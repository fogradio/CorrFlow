#!/usr/bin/env python
"""CMMD — MMD between CLIP embeddings (Jayasumana et al., "Rethinking FID", CVPR 2024).

    CMMD = SCALE * ( E k(x,x') + E k(y,y') - 2 E k(x,y) ),   k = Gaussian RBF

Unlike FID it assumes no Gaussian form for the feature distribution, is far more
sample-efficient, and rides on CLIP rather than Inception features.

Follows the reference implementation's conventions so the numbers are comparable to
published CMMD:
  - CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) `get_image_features`, L2-normalized
  - images resized to 224 bicubic, values in [0,1], CLIP mean/std normalization
  - sigma = 10  (gamma = 1/(2*sigma^2)),  scale = 1000
  - V-statistic form: kernel means include the diagonal

Both feature extraction and the kernel means run in fp32 with TF32 disabled and
float64 accumulation -- CMMD is a ~1e-4 relative difference between kernel means
near 0.99, so it is unusually exposed to cancellation.
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_fdd import _ArrDataset, load_images  # noqa: E402

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_SIGMA = 10.0
_SCALE = 1000.0


class ClipEmbedder(torch.nn.Module):
    def __init__(self, model_id="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        from transformers import CLIPModel

        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.model.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1).to(device))
        self.input_size = self.model.config.vision_config.image_size
        self.embed_dim = self.model.config.projection_dim

    @torch.inference_mode()
    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.float() / 255.0
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bicubic", align_corners=False, antialias=True)
        x = x.clamp(0, 1)
        x = (x - self.mean) / self.std
        emb = self.model.get_image_features(pixel_values=x)
        return F.normalize(emb.float(), dim=-1)


@torch.inference_mode()
def extract(embedder, arr, batch_size, device, num_workers, limit=None, tag=""):
    if limit is not None and arr.shape[0] > limit:
        arr = arr[:limit]
    n = arr.shape[0]
    loader = torch.utils.data.DataLoader(
        _ArrDataset(arr), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    out = torch.empty(n, embedder.embed_dim, dtype=torch.float32)
    t0, seen = time.time(), 0
    for x in loader:
        e = embedder(x.to(device, non_blocking=True)).cpu()
        out[seen:seen + e.shape[0]] = e
        seen += e.shape[0]
        if seen % (batch_size * 60) == 0 or seen == n:
            print(f"    {tag} {seen}/{n}  {seen/(time.time()-t0):.0f} img/s", flush=True)
    return out


def cached(embedder, path, cache_dir, batch_size, device, num_workers, limit):
    stem = Path(path).stem
    cache = Path(cache_dir) / f"{stem}.clipL14_proj.f32.npy" if cache_dir else None
    if cache is not None and cache.exists():
        e = torch.from_numpy(np.load(cache))
        print(f"  cached embeddings {tuple(e.shape)} <- {cache.name}")
        return e
    arr = load_images(path)
    e = extract(embedder, arr, batch_size, device, num_workers, limit, stem)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, e.numpy())
    del arr
    return e


@torch.inference_mode()
def _kernel_mean(A: torch.Tensor, B: torch.Tensor, gamma: float, row_bs: int = 4096) -> float:
    """float64-accumulated mean of exp(-gamma * ||a-b||^2) over all pairs."""
    a2 = (A * A).sum(1)
    b2 = (B * B).sum(1)
    total = torch.zeros((), dtype=torch.float64, device=A.device)
    for i in range(0, A.shape[0], row_bs):
        d = a2[i:i + row_bs, None] + b2[None, :] - 2.0 * (A[i:i + row_bs] @ B.T)
        d.clamp_min_(0)
        total += torch.exp(-gamma * d).sum(dtype=torch.float64)
    return float(total / (A.shape[0] * B.shape[0]))


def cmmd(x: torch.Tensor, y: torch.Tensor, row_bs: int = 4096, k_xx: float | None = None):
    gamma = 1.0 / (2.0 * _SIGMA ** 2)
    if k_xx is None:
        k_xx = _kernel_mean(x, x, gamma, row_bs)
    k_yy = _kernel_mean(y, y, gamma, row_bs)
    k_xy = _kernel_mean(x, y, gamma, row_bs)
    return _SCALE * (k_xx + k_yy - 2 * k_xy), k_xx


def main():
    ap = argparse.ArgumentParser(description="CMMD (CLIP-MMD) for generated npz sets")
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--reference-images", required=True)
    ap.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--row-bs", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature-cache-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False

    paths = []
    for p in args.npz:
        paths += sorted(str(q) for q in Path(p).glob("*.npz")) if os.path.isdir(p) else [p]

    device = torch.device(args.device)
    emb = ClipEmbedder(args.model_id, device)
    print(f"model     : {args.model_id}  ({emb.input_size}px, {emb.embed_dim}-d projected, L2-normed)")
    print(f"kernel    : gaussian, sigma={_SIGMA}, scale={_SCALE}")
    print(f"real ref  : {args.reference_images}")

    print(f"\n=== reference: {Path(args.reference_images).stem} ===")
    ref = cached(emb, args.reference_images, args.feature_cache_dir,
                 args.batch_size, device, args.num_workers, args.num_samples).to(device)

    results, k_xx = {}, None
    for path in paths:
        stem = Path(path).stem
        print(f"\n=== {stem} ===", flush=True)
        gen = cached(emb, path, args.feature_cache_dir, args.batch_size,
                     device, args.num_workers, args.num_samples).to(device)
        t0 = time.time()
        val, k_xx = cmmd(ref, gen, args.row_bs, k_xx)
        results[stem] = {"n": int(gen.shape[0]), "cmmd": val}
        print(f"  CMMD = {val:.4f}   ({time.time()-t0:.0f}s)", flush=True)
        del gen
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"{'config':<50} {'CMMD ↓':>12}")
    print("=" * 70)
    for k, v in results.items():
        print(f"{k:<50} {v['cmmd']:>12.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"model": args.model_id, "sigma": _SIGMA, "scale": _SCALE,
                       "reference_images": args.reference_images, "results": results}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
