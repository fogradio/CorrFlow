#!/usr/bin/env python
"""FD-DINOv2 (FDD) on generated uint8 NHWC npz sets.

Same Frechet formula as FID, but features come from DINOv2 ViT-L/14 (CLS token,
256x256 input) instead of Inception-v3 -- the representation Stein et al. found
to align best with human judgement of generative quality.

Reuses fd_evaluator's extractor / accumulator / reference moments so the numbers
are directly comparable to the FID already reported by offline_eval.py. The
packaged `fdr_dinov2_cls` metric only exposes FD / norm_const; this reports the
raw FD (what papers call FDD / FD-DINOv2) alongside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from fd_evaluator.api import _load_catalogue, _spec_for
from fd_evaluator.extractors import build_extractor
from fd_evaluator.frechet import FrechetAccumulator, frechet_distance
from fd_evaluator.stats import load_moments


def mmap_npz_array(path: str) -> np.ndarray:
    """Memory-map the single stored (uncompressed) array inside an .npz.

    np.load(mmap_mode=...) is a no-op for .npz, so a plain load would pull all
    ~10GB per file into RAM. These npz are written with np.savez (ZIP_STORED),
    so the payload can be mapped in place.
    """
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        name = "arr_0.npy" if "arr_0.npy" in names else names[0]
        info = zf.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("compressed npz; cannot mmap")
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        lh = fh.read(30)
        n_name = int.from_bytes(lh[26:28], "little")
        n_extra = int.from_bytes(lh[28:30], "little")
        fh.seek(info.header_offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        else:
            raise ValueError(f"unsupported npy version {version}")
        if fortran:
            raise ValueError("fortran-order array not supported")
        data_offset = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=shape)


def load_images(path: str) -> np.ndarray:
    try:
        arr = mmap_npz_array(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] mmap failed ({e}); falling back to full np.load")
        data = np.load(path)
        key = "arr_0" if "arr_0" in data else list(data.keys())[0]
        arr = data[key]
    if arr.dtype != np.uint8 or arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected uint8 NHWC RGB, got dtype={arr.dtype} shape={arr.shape}")
    return arr


class _ArrDataset(torch.utils.data.Dataset):
    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.arr[i])).permute(2, 0, 1)


@torch.inference_mode()
def features_moments(extractor, arr, batch_size, device, num_workers, limit=None,
                     save_features=None, tag=""):
    if limit is not None and arr.shape[0] > limit:
        arr = arr[:limit]
    n = arr.shape[0]
    loader = torch.utils.data.DataLoader(
        _ArrDataset(arr), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    acc = FrechetAccumulator(extractor.spec.feat_dim)
    kept = [] if save_features else None
    t0 = time.time()
    seen = 0
    for x in loader:
        f = extractor(x.to(device, non_blocking=True)).detach().cpu()
        acc.update(f)
        if kept is not None:
            kept.append(f)
        seen += x.shape[0]
        if seen % (batch_size * 40) == 0 or seen == n:
            el = time.time() - t0
            print(f"    {tag} {seen}/{n}  {seen/el:.0f} img/s  eta {(n-seen)/(seen/el):.0f}s",
                  flush=True)
    mu, sigma = acc.finalize()
    if kept is not None:
        os.makedirs(os.path.dirname(save_features), exist_ok=True)
        torch.save(torch.cat(kept, 0), save_features)
    return mu, sigma, n


def main():
    ap = argparse.ArgumentParser(description="FD-DINOv2 (FDD) for generated npz sets")
    ap.add_argument("npz", nargs="+", help="generated uint8 NHWC .npz files (or a directory)")
    ap.add_argument("--extractor", default="dinov2_cls",
                    help="fd_evaluator extractor name (default: dinov2_cls = DINOv2 ViT-L/14)")
    ap.add_argument("--reference", default=None,
                    help="reference moments name or .npz path (default: extractor's default)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=None, help="truncate each set to N images")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    ap.add_argument("--save-moments-dir", default=None,
                    help="cache per-set mu/sigma npz here (skips recompute on re-run)")
    ap.add_argument("--save-features-dir", default=None,
                    help="also dump raw features (.pt) here, for Precision/Recall later")
    args = ap.parse_args()

    paths = []
    for p in args.npz:
        if os.path.isdir(p):
            paths += sorted(str(q) for q in Path(p).glob("*.npz"))
        else:
            paths.append(p)
    if not paths:
        sys.exit("no npz found")

    spec = _spec_for(args.extractor, _load_catalogue())
    ref_name = args.reference or spec.default_reference
    print(f"extractor : {args.extractor} ({spec.model}, {spec.input_size}px, pool={spec.pool})")
    print(f"reference : {ref_name}")
    mu_ref, sig_ref = load_moments(ref_name)

    device = torch.device(args.device)
    extractor = build_extractor(spec).to(device).eval()

    results = {}
    for path in paths:
        name = Path(path).stem
        print(f"\n=== {name} ===", flush=True)
        cache = (Path(args.save_moments_dir) / f"{name}.{args.extractor}.moments.npz"
                 if args.save_moments_dir else None)
        if cache is not None and cache.exists():
            d = np.load(cache)
            mu, sigma, n = d["mu"], d["sigma"], int(d["n"])
            print(f"  using cached moments ({n} samples)")
        else:
            arr = load_images(path)
            print(f"  shape={arr.shape} dtype={arr.dtype}")
            feat_path = (str(Path(args.save_features_dir) / f"{name}.{args.extractor}.pt")
                         if args.save_features_dir else None)
            mu, sigma, n = features_moments(
                extractor, arr, args.batch_size, device, args.num_workers,
                limit=args.num_samples, save_features=feat_path, tag=name)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, mu=mu, sigma=sigma, n=n)
            del arr

        fd = frechet_distance(mu, sigma, mu_ref, sig_ref)
        row = {"n": n, f"fd_{args.extractor}": fd}
        if spec.norm_const > 0:
            row[f"fdr_{args.extractor}"] = fd / spec.norm_const
        results[name] = row
        print(f"  FD ({args.extractor}) = {fd:.4f}"
              + (f"   [FDr = {fd/spec.norm_const:.4f}]" if spec.norm_const > 0 else ""))

    print("\n" + "=" * 72)
    print(f"{'config':<50} {'N':>7} {'FDD':>10}")
    print("=" * 72)
    for k, v in results.items():
        print(f"{k:<50} {v['n']:>7} {v[f'fd_{args.extractor}']:>10.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"extractor": args.extractor, "reference": ref_name,
                       "results": results}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
