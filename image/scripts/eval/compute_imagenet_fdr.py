#!/usr/bin/env python
"""Evaluate saved ImageNet NPZs with the installed, historically used FD protocol.

Uses compute_fdd's read-only NPZ mmap and streaming float64 moments, and
fd_evaluator's exact extractors, preprocessing, references, and FD formula.
One backbone is held on one device at a time. Each completed pair is persisted
as JSON, with input/protocol fingerprints so a later job can resume safely.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from compute_fdd import features_moments, mmap_npz_array
from fd_evaluator.api import _load_catalogue, _spec_for
from fd_evaluator.extractors import build_extractor
from fd_evaluator.frechet import frechet_distance
from fd_evaluator.stats import load_moments

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "results/stage2/offline_eval/imagenet_missing_fdr"
INCEPTION_WEIGHT = "weights-inception-2015-12-05-6726825d.pth"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    """Publish our own output atomically, preserving completed work on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f"{path.name}.writing-{os.getpid()}")
    with staging.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(staging, path)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def inspect_npz(path, expected_shape):
    arr = mmap_npz_array(str(path))  # No full-array fallback for these 9.8 GB files.
    if arr.dtype != np.uint8 or list(arr.shape) != expected_shape:
        raise ValueError(f"{path}: expected uint8 {expected_shape}, got {arr.dtype} {arr.shape}")
    if arr.offset + arr.nbytes > Path(path).stat().st_size:
        raise ValueError(f"truncated image payload: {path}")
    identity = file_identity(path)
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo("arr_0.npy" if "arr_0.npy" in archive.namelist()
                                 else archive.namelist()[0])
        identity.update(member=member.filename, member_crc32=member.CRC,
                        member_bytes=member.file_size)
    identity.update(shape=list(arr.shape), dtype=str(arr.dtype))
    return identity


def source_csv(path):
    path = path.parent.parent / f"{path.stem}.csv"
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {"path": str(path), "rows": rows, "usage": "Historical values only, not recomputed here."}


def read_inputs(args, catalogue):
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["rows"]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_+.-]+", x) for x in ids):
        raise ValueError("manifest row IDs must be unique safe filenames")
    if args.rows:
        unknown = set(args.rows) - set(ids)
        if unknown:
            raise ValueError(f"unknown row IDs: {sorted(unknown)}")
        rows = [row for row in rows if row["id"] in args.rows]
    if not rows:
        raise ValueError("no input rows selected")
    selected = []
    for row in rows:
        row = dict(row)
        path = (args.npz_dir / row["npz"]).resolve()
        row["input"] = inspect_npz(path, manifest["expected_shape"])
        row["source_csv"] = source_csv(path)
        requested = args.extractors or (manifest["extractors"] + row.get("additional_extractors", []))
        row["extractors"] = list(dict.fromkeys(requested))
        if not row["extractors"] or set(row["extractors"]) - set(catalogue["extractors"]):
            raise ValueError(f"unknown/empty extractors for {row['id']}: {row['extractors']}")
        selected.append(row)
    return manifest, selected


def reference_path(spec, catalogue, args, revision, download=False):
    """Reuse the old cache, including its incomplete stats-flat, then download."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    filename = catalogue["references"][spec.default_reference]
    roots = [args.asset_dir / "stats"]
    if os.environ.get("NANOGEN_EVALS_STATS_DIR"):
        roots.append(Path(os.environ["NANOGEN_EVALS_STATS_DIR"]))
    roots.append(Path.home() / ".cache/nanogen-evals/stats-flat")
    for root in roots:
        path = root / filename
        if path.is_file():
            # fd_evaluator.load_moments recognizes paths by the .npz suffix.
            # Keep HF snapshot symlinks instead of resolving to extensionless blobs.
            return path.absolute()
    cache_roots = [args.asset_dir / "hf_stats", Path.home() / ".cache/nanogen-evals/stats"]
    for root in cache_roots:
        path = try_to_load_from_cache(catalogue["hf_repo_id"], filename, cache_dir=str(root),
                                     repo_type="dataset", revision=revision)
        if isinstance(path, str) and Path(path).is_file():
            return Path(path).absolute()
    if not download:
        return None
    return Path(hf_hub_download(
        repo_id=catalogue["hf_repo_id"], filename=filename, repo_type="dataset",
        revision=revision, cache_dir=str(cache_roots[0]),
    )).absolute()


def weight_path(spec):
    if spec.backend == "torch_fidelity":
        path = Path(torch.hub.get_dir()) / "checkpoints" / INCEPTION_WEIGHT
        return path.resolve() if path.is_file() else None
    from huggingface_hub import try_to_load_from_cache
    from timm.models import get_pretrained_cfg

    cfg = get_pretrained_cfg(spec.model)
    for filename in [cfg.hf_hub_filename or "model.safetensors", "pytorch_model.bin"]:
        path = try_to_load_from_cache(cfg.hf_hub_id, filename)
        if isinstance(path, str) and Path(path).is_file():
            return Path(path).resolve()
    return None


def runtime_protocol():
    import fd_evaluator

    package = Path(fd_evaluator.__file__).parent
    paths = [package / name for name in ("catalogue.yaml", "extractors.py", "frechet.py")]
    paths += [Path(__file__), Path(__file__).with_name("compute_fdd.py")]
    return {
        "source_sha256": {str(p): sha256(p) for p in paths},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("fd-evaluator", "torch", "torchvision", "timm", "torch-fidelity", "numpy", "scipy")},
        "tf32": True,
        "precision": "fd_evaluator defaults: Inception/ConvNeXt fp32, timm CLS backbone CUDA bf16 autocast",
        "moments": "float64 streaming, unbiased covariance (N-1)",
        "images": "all saved uint8 RGB images in original order, no new sampling or truncation",
        "fdr_definition": "fd_<extractor> / catalogue norm_const",
        "inception_reference": "guided_diffusion_stats (ADM), distinct from the FID column's imagenet_256_fid_stats",
    }


def validate_result(result, signature, name, expected_n):
    if result["signature"] != signature or result["n"] != expected_n:
        raise ValueError("cached pair does not match input/protocol/sample count")
    fd, fdr = result["metrics"][f"fd_{name}"], result["metrics"][f"fdr_{name}"]
    norm = signature["extractor"]["norm_const"]
    if not all(math.isfinite(x) for x in (fd, fdr)) or not math.isclose(fd / norm, fdr, abs_tol=1e-12):
        raise ValueError("invalid cached FD/FDr values")


def evaluate_pair(extractor, row, ref_moments, signature, args):
    name = extractor.spec.name
    digest = json_digest(signature)
    path = args.out_dir / "pairs" / f"{row['id']}.{name}.{digest[:20]}.json"
    n = row["input"]["shape"][0]
    if path.is_file():
        result = json.loads(path.read_text())
        validate_result(result, signature, name, n)
        print(f"[resume] {row['id']} {name}", flush=True)
        return result, path
    started = time.monotonic()
    arr = mmap_npz_array(row["input"]["path"])
    mu, sigma, seen = features_moments(
        extractor, arr, args.batch_size, torch.device(args.device), args.num_workers,
        tag=f"{row['id']}/{name}",
    )
    if seen != n or inspect_npz(row["input"]["path"], list(arr.shape)) != row["input"]:
        raise ValueError("input changed during evaluation or sample count mismatch")
    fd = float(frechet_distance(mu, sigma, *ref_moments))
    result = {
        "signature": signature, "n": seen,
        "metrics": {f"fd_{name}": fd, f"fdr_{name}": fd / extractor.spec.norm_const},
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_result(result, signature, name, n)
    write_json(path, result)
    print(f"[result] {row['id']} {name} FD={fd:.8f} FDr={fd/extractor.spec.norm_const:.8f}", flush=True)
    return result, path


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).with_name("imagenet_missing_fdr_manifest.json"))
    ap.add_argument("--npz-dir", type=Path, default=REPO / "results/stage2/offline_eval/gen_npz")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--asset-dir", type=Path, default=DEFAULT_OUTPUT / "assets")
    ap.add_argument("--rows", nargs="+", help="manifest row IDs to evaluate")
    ap.add_argument("--extractors", nargs="+", help="override all per-row extractors, including DINOv2 rechecks")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="validate all NPZ headers and inspect local assets, no downloads/GPU")
    modes.add_argument("--prepare-only", action="store_true", help="download references/weights and load every extractor on CPU")
    modes.add_argument("--smoke-only", action="store_true", help="two-image forward per extractor, no paper metrics")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive, num workers nonnegative")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cat = _load_catalogue()
    manifest, rows = read_inputs(args, cat)
    ext_names = [x for x in cat["extractors"] if any(x in row["extractors"] for row in rows)]
    specs = {name: _spec_for(name, cat) for name in ext_names}
    protocol = runtime_protocol()
    mode = "dry_run" if args.dry_run else "prepare" if args.prepare_only else "smoke" if args.smoke_only else "eval"
    report = {
        "schema_version": 1, "mode": mode, "status": "in_progress",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "notes": manifest["notes"], "protocol": protocol,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "planned_pairs": sum(len(row["extractors"]) for row in rows),
        "assets": {}, "results": {},
    }
    for row in rows:
        report["results"][row["id"]] = {
            **row, "n": row["input"]["shape"][0], "metrics": {}, "pair_files": {},
        }
    for name, spec in specs.items():
        ref = reference_path(spec, cat, args, manifest["stats_revision"])
        weight = weight_path(spec)
        report["assets"][name] = {
            "extractor": asdict(spec), "reference_file": str(ref) if ref else None,
            "weight_file": str(weight) if weight else None,
        }
    print(f"[plan] {len(rows)} NPZs, {report['planned_pairs']} pairs, mode={mode}", flush=True)
    output = args.out_dir / ({"dry_run": "preflight.json", "prepare": "prepared_assets.json",
                             "smoke": "smoke.json", "eval": "imagenet_fdr.json"}[mode])
    if args.dry_run:
        missing = [name for name, a in report["assets"].items() if not a["reference_file"] or not a["weight_file"]]
        report["status"] = "needs_assets" if missing else "ready"
        report["missing_assets"] = missing
        write_json(output, report)
        print(f"[preflight] NPZs valid. Missing assets: {missing}. Report: {output}")
        return
    if not args.prepare_only and torch.device(args.device).type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Use --dry-run/--prepare-only on a CPU host.")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Evaluation requires exactly one visible GPU. Select it with CUDA_VISIBLE_DEVICES.")
        torch.cuda.set_device(torch.device(args.device))
        report["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    write_json(output, report)
    try:
        # Resolve and validate ALL reference moments before any 50K feature pass.
        references = {}
        for name, spec in specs.items():
            ref = reference_path(spec, cat, args, manifest["stats_revision"], download=True)
            moments = load_moments(str(ref))
            if moments[0].shape != (spec.feat_dim,) or moments[1].shape != (spec.feat_dim, spec.feat_dim):
                raise ValueError(f"reference dimension mismatch for {name}: {ref}")
            if not all(np.isfinite(m).all() for m in moments):
                raise ValueError(f"nonfinite reference moments: {ref}")
            references[name] = moments
            report["assets"][name].update(reference_file=str(ref), reference_sha256=sha256(ref))
        for name, spec in specs.items():
            print(f"[model] {name}: {spec.model}", flush=True)
            extractor = build_extractor(spec).eval()
            weight = weight_path(spec)
            if weight is None:
                raise RuntimeError(f"Cannot identify cached weights loaded for {name}")
            report["assets"][name]["weight_file"] = str(weight)
            if args.prepare_only:
                report["assets"][name]["loaded_on_cpu"] = True
            else:
                extractor = extractor.to(args.device)
                if args.smoke_only:
                    row = next(row for row in rows if name in row["extractors"])
                    arr = mmap_npz_array(row["input"]["path"])
                    batch = torch.from_numpy(np.array(arr[:2])).permute(0, 3, 1, 2).contiguous().to(args.device)
                    with torch.inference_mode():
                        feats = extractor(batch)
                    if tuple(feats.shape) != (2, spec.feat_dim) or not torch.isfinite(feats).all():
                        raise ValueError(f"invalid smoke features: {name} {feats.shape}")
                    report["assets"][name]["smoke_feature_shape"] = list(feats.shape)
                    del batch, feats, arr
                else:
                    for row in rows:
                        if name not in row["extractors"]:
                            continue
                        signature = {
                            "input": row["input"], "extractor": asdict(spec), "protocol": protocol,
                            "reference_sha256": report["assets"][name]["reference_sha256"],
                            "weight": file_identity(weight), "batch_size": args.batch_size,
                            "device_type": torch.device(args.device).type,
                        }
                        result, pair_path = evaluate_pair(extractor, row, references[name], signature, args)
                        report["results"][row["id"]]["metrics"].update(result["metrics"])
                        report["results"][row["id"]]["pair_files"][name] = str(pair_path)
                        write_json(output, report)
            del extractor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_json(output, report)
        if mode == "eval":
            for row in rows:
                values = report["results"][row["id"]]["metrics"]
                if any(f"fdr_{name}" not in values for name in row["extractors"]):
                    raise RuntimeError(f"incomplete metrics: {row['id']}")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output, report)
        raise
    write_json(output, report)
    print(f"[done] {mode}: {output}", flush=True)


if __name__ == "__main__":
    main()
