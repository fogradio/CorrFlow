#!/usr/bin/env python
"""Improved Precision / Recall (Kynkaanniemi et al. 2019) for generated npz sets.

Precision = fraction of GENERATED samples that fall inside the real manifold.
            -> fidelity: are the samples plausible?
Recall    = fraction of REAL samples that fall inside the generated manifold.
            -> coverage: does the model cover the real distribution?

Each manifold is the union of hyperspheres around every sample, with radius set
to that sample's k-th nearest neighbour within its own set (k=3 by default).

Feature space is pluggable via the same fd_evaluator catalogue used by
compute_fdd.py, so P/R can be reported in Inception space (comparable to the
ADM/DiT/SiT convention) or DINOv2 space (consistent with FDD).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_fdd import _ArrDataset, load_images  # noqa: E402

from fd_evaluator.api import _load_catalogue, _spec_for  # noqa: E402
from fd_evaluator.extractors import build_extractor  # noqa: E402
from fd_evaluator.frechet import frechet_distance  # noqa: E402
from fd_evaluator.stats import load_moments  # noqa: E402


@torch.inference_mode()
def extract_features(extractor, arr, batch_size, device, num_workers, limit=None, tag=""):
    if limit is not None and arr.shape[0] > limit:
        arr = arr[:limit]
    n = arr.shape[0]
    loader = torch.utils.data.DataLoader(
        _ArrDataset(arr), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    out = torch.empty(n, extractor.spec.feat_dim, dtype=torch.float32)
    t0, seen = time.time(), 0
    for x in loader:
        f = extractor(x.to(device, non_blocking=True)).float().cpu()
        out[seen:seen + f.shape[0]] = f
        seen += f.shape[0]
        if seen % (batch_size * 60) == 0 or seen == n:
            el = time.time() - t0
            print(f"    {tag} {seen}/{n}  {seen/el:.0f} img/s", flush=True)
    return out


def cached_features(extractor, path, cache_dir, batch_size, device, num_workers, limit):
    name = Path(path).stem
    cache = Path(cache_dir) / f"{name}.{extractor.spec.name}.f32.npy" if cache_dir else None
    if cache is not None and cache.exists():
        feats = torch.from_numpy(np.load(cache))
        print(f"  cached features {tuple(feats.shape)} <- {cache.name}")
        return feats
    arr = load_images(path)
    print(f"  shape={arr.shape}")
    feats = extract_features(extractor, arr, batch_size, device, num_workers, limit, name)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, feats.numpy())
    del arr
    return feats


@torch.inference_mode()
def _knn_radii_sq(X: torch.Tensor, k: int, row_bs: int = 2048) -> torch.Tensor:
    """Squared distance to each row's k-th nearest neighbour within X (self excluded)."""
    n = X.shape[0]
    x2 = (X * X).sum(1)
    out = torch.empty(n, device=X.device)
    for i in range(0, n, row_bs):
        d = x2[i:i + row_bs, None] + x2[None, :] - 2.0 * (X[i:i + row_bs] @ X.T)
        d.clamp_min_(0)
        # k+1 smallest includes the self-distance (0), so index k is the k-th NN
        out[i:i + row_bs] = torch.topk(d, k + 1, dim=1, largest=False).values[:, k]
    return out


@torch.inference_mode()
def precision_recall(real: torch.Tensor, gen: torch.Tensor, k: int, row_bs: int = 2048):
    """One pass over the gen x real distance matrix serves both directions."""
    r_rad2 = _knn_radii_sq(real, k, row_bs)
    g_rad2 = _knn_radii_sq(gen, k, row_bs)
    r2 = (real * real).sum(1)
    g2 = (gen * gen).sum(1)
    prec_hit = torch.zeros(gen.shape[0], dtype=torch.bool, device=gen.device)
    rec_hit = torch.zeros(real.shape[0], dtype=torch.bool, device=real.device)
    for i in range(0, gen.shape[0], row_bs):
        d = g2[i:i + row_bs, None] + r2[None, :] - 2.0 * (gen[i:i + row_bs] @ real.T)
        d.clamp_min_(0)
        prec_hit[i:i + row_bs] = (d <= r_rad2[None, :]).any(dim=1)
        rec_hit |= (d <= g_rad2[i:i + row_bs, None]).any(dim=0)
    return prec_hit.float().mean().item(), rec_hit.float().mean().item()


def main():
    ap = argparse.ArgumentParser(description="Improved Precision/Recall for generated npz sets")
    ap.add_argument("npz", nargs="+", help="generated uint8 NHWC .npz files (or a directory)")
    ap.add_argument("--reference-images", required=True,
                    help="real uint8 NHWC .npz defining the real manifold")
    ap.add_argument("--extractor", default="dinov2_cls")
    ap.add_argument("-k", "--nhood-size", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--row-bs", type=int, default=2048, help="rows per distance-matrix block")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature-cache-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--also-fd", action="store_true",
                    help="also report FD in this feature space from the same features")
    args = ap.parse_args()

    # TF32 has a 10-bit mantissa; the k-NN radius comparisons sit right at the
    # decision boundary, so keep the distance matmuls in true fp32.
    torch.backends.cuda.matmul.allow_tf32 = False

    paths = []
    for p in args.npz:
        paths += sorted(str(q) for q in Path(p).glob("*.npz")) if os.path.isdir(p) else [p]
    if not paths:
        sys.exit("no npz found")

    spec = _spec_for(args.extractor, _load_catalogue())
    device = torch.device(args.device)
    print(f"extractor : {args.extractor} ({spec.model}, {spec.input_size}px, pool={spec.pool})")
    print(f"real ref  : {args.reference_images}")
    print(f"k         : {args.nhood_size}")

    extractor = build_extractor(spec).to(device).eval()

    print(f"\n=== reference: {Path(args.reference_images).stem} ===")
    real = cached_features(extractor, args.reference_images, args.feature_cache_dir,
                           args.batch_size, device, args.num_workers, args.num_samples)
    real_g = real.to(device)

    mu_ref = sig_ref = None
    if args.also_fd:
        mu_ref, sig_ref = load_moments(spec.default_reference)

    results = {}
    for path in paths:
        name = Path(path).stem
        print(f"\n=== {name} ===", flush=True)
        gen = cached_features(extractor, path, args.feature_cache_dir,
                              args.batch_size, device, args.num_workers, args.num_samples)
        gen_g = gen.to(device)
        t0 = time.time()
        p, r = precision_recall(real_g, gen_g, args.nhood_size, args.row_bs)
        row = {"n": int(gen.shape[0]), "precision": p, "recall": r,
               "f1": (2 * p * r / (p + r)) if (p + r) > 0 else 0.0}
        if args.also_fd:
            f = gen.double().numpy()
            mu = f.mean(0)
            fc = f - mu
            sigma = fc.T @ fc / (f.shape[0] - 1)
            row[f"fd_{args.extractor}"] = frechet_distance(mu, sigma, mu_ref, sig_ref)
        results[name] = row
        extra = (f"  FD={row[f'fd_{args.extractor}']:.4f}" if args.also_fd else "")
        print(f"  Precision = {p:.4f}   Recall = {r:.4f}   F1 = {row['f1']:.4f}"
              f"{extra}   ({time.time()-t0:.0f}s)", flush=True)
        del gen_g, gen
        torch.cuda.empty_cache()

    print("\n" + "=" * 84)
    print(f"{'config':<48} {'N':>7} {'Prec↑':>8} {'Recall↑':>8} {'F1↑':>8}")
    print("=" * 84)
    for kk, v in results.items():
        print(f"{kk:<48} {v['n']:>7} {v['precision']:>8.4f} {v['recall']:>8.4f} {v['f1']:>8.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"extractor": args.extractor, "k": args.nhood_size,
                       "reference_images": args.reference_images,
                       "results": results}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
