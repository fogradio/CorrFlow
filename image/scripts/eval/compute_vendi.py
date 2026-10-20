#!/usr/bin/env python
"""Per-class Vendi Score on DINOv2 features — intra-class diversity diagnostic.

Global Vendi on ImageNet mostly measures INTER-class spread, which is forced to be
uniform by construction (50 samples per class), so it carries little signal. The
informative quantity is diversity WITHIN each class, averaged over classes:

    V_c = Vendi({f(x_i) : y_i = c}),    Vendi_class = mean_c V_c

Vendi Score (Friedman & Dieng 2023) with a cosine-similarity kernel: for the n x n
kernel matrix K (K_ii = 1), take eigenvalues lambda of K/n and return
exp(-sum lambda log lambda) — the effective number of distinct modes, in [1, n].

This catches the failure mode FID and Precision are blind to: a guided/refined model
that always renders "golden retriever" as front-facing, centred, on grass scores well
on fidelity while its per-class distribution has collapsed.

Consumes the float32 feature caches written by compute_prd.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def vendi_from_features(X: torch.Tensor, eps: float = 1e-12) -> float:
    """Vendi score with cosine kernel. X: [n, d] float64 on any device."""
    n = X.shape[0]
    Xn = torch.nn.functional.normalize(X, dim=1)
    K = (Xn @ Xn.T) / n
    lam = torch.linalg.eigvalsh(K)
    lam = lam.clamp_min(0)
    lam = lam / lam.sum()
    nz = lam[lam > eps]
    entropy = -(nz * nz.log()).sum()
    return float(torch.exp(entropy))


def per_class_vendi(feats: torch.Tensor, labels: np.ndarray, num_classes: int, device):
    vs = np.empty(num_classes, dtype=np.float64)
    counts = np.empty(num_classes, dtype=np.int64)
    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        counts[c] = len(idx)
        Xc = feats[idx].to(device=device, dtype=torch.float64)
        vs[c] = vendi_from_features(Xc)
    return vs, counts


def main():
    ap = argparse.ArgumentParser(description="Per-class Vendi score on cached features")
    ap.add_argument("--feature-dir", default="results/stage2/offline_eval/feats")
    ap.add_argument("--extractor", default="dinov2_cls")
    ap.add_argument("--sets", nargs="+", required=True,
                    help="feature-cache stems, e.g. in256-k7-...-ep80_ema")
    ap.add_argument("--gen-labels", default=None,
                    help="labels npy aligned with the generated npz row order")
    ap.add_argument("--real-set", default="imagenet-256-val",
                    help="stem of the real reference feature cache (baseline row)")
    ap.add_argument("--real-labels", default=None, help="labels npy for --real-set")
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    fdir = Path(args.feature_dir)

    def load(stem):
        p = fdir / f"{stem}.{args.extractor}.f32.npy"
        if not p.exists():
            raise FileNotFoundError(f"{p} — run compute_prd.py first to build the cache")
        return torch.from_numpy(np.load(p))

    jobs = []
    if args.real_set:
        jobs.append((args.real_set, args.real_labels or args.gen_labels, True))
    jobs += [(s, args.gen_labels, False) for s in args.sets]

    results = {}
    print(f"feature space: {args.extractor}   classes: {args.num_classes}\n")
    for stem, lab_path, is_real in jobs:
        feats = load(stem)
        labels = np.load(lab_path)
        if len(labels) != feats.shape[0]:
            raise ValueError(f"{stem}: {feats.shape[0]} features vs {len(labels)} labels")
        vs, counts = per_class_vendi(feats, labels, args.num_classes, device)
        row = {
            "n": int(feats.shape[0]),
            "samples_per_class": int(counts.min()) if counts.min() == counts.max() else None,
            "vendi_class_mean": float(vs.mean()),
            "vendi_class_std": float(vs.std()),
            "vendi_class_min": float(vs.min()),
            "vendi_class_max": float(vs.max()),
            "is_real_reference": is_real,
        }
        results[stem] = row
        tag = "  [real]" if is_real else ""
        print(f"{stem:<48} Vendi_class = {vs.mean():8.4f} "
              f"(sd {vs.std():.3f}, min {vs.min():.2f}, max {vs.max():.2f}){tag}", flush=True)
        results[stem]["_per_class"] = vs.tolist()

    real = results.get(args.real_set)
    print("\n" + "=" * 78)
    print(f"{'config':<48} {'Vendi_class↑':>13} {'% of real':>11}")
    print("=" * 78)
    for k, v in results.items():
        pct = f"{100*v['vendi_class_mean']/real['vendi_class_mean']:.2f}%" if real else "-"
        print(f"{k:<48} {v['vendi_class_mean']:>13.4f} {pct:>11}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"extractor": args.extractor, "num_classes": args.num_classes,
                       "results": results}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
