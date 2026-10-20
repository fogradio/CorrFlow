"""Sampling-time refiner correction.

Attached to ``Transport.refiner_runner``; the drift adds ``scale * refiner(...)``
to the velocity **after** CFG/IG have been combined, so there is no
``cfg_scale * refiner_scale`` amplification (the failure mode documented in the
reference's inference script).
"""

from __future__ import annotations

import torch

from .capture import DiTHiddenCapture


class RefinerRunner:
    """Callable returning the velocity correction for one ODE step."""

    def __init__(self, refiner, capture, sampling_config, *, use_guidance: bool,
                 input_v_prediction: bool = False, t_eps: float = 0.05):
        self.refiner = refiner
        self.capture = capture
        self.cfg = sampling_config
        self.use_guidance = bool(use_guidance)
        self.input_v_prediction = bool(input_v_prediction)
        self.t_eps = float(t_eps)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: torch.Tensor, x_pred: torch.Tensor,
                 velocity: torch.Tensor) -> torch.Tensor:
        if self.cfg.scale == 0.0:
            return torch.zeros_like(velocity)

        # Under guidance the sampler feeds a [cond, uncond] doubled batch and the
        # guidance helpers duplicate their output; only the first half is unique.
        total = velocity.shape[0]
        n = total // 2 if self.use_guidance else total
        payload = DiTHiddenCapture.slice_payload(self.capture.finalize(), n)

        t_n = t[:n]
        t_safe = t_n.view(-1, *([1] * (x.ndim - 1))).clamp_min(self.t_eps)
        x_pred = x_pred[:n]
        refiner_input = (x[:n] - x_pred) / t_safe if self.input_v_prediction else x_pred

        delta = self.refiner(
            refiner_input, payload["t_emb"], payload["cond_emb"], payload["hidden"]
        ).to(velocity.dtype) * self.cfg.scale

        band = ((t_n >= self.cfg.t_min) & (t_n <= self.cfg.t_max))
        delta = torch.where(band.view(-1, *([1] * (delta.ndim - 1))), delta, torch.zeros_like(delta))

        if self.cfg.trust_rho > 0:
            v_rms = velocity[:n].float().pow(2).mean().sqrt()
            d_rms = delta.float().pow(2).mean().sqrt()
            factor = torch.clamp(self.cfg.trust_rho * v_rms / (d_rms + 1e-8), max=1.0)
            delta = delta * factor.to(delta.dtype)

        if n != total:
            delta = delta.repeat(total // n, *([1] * (delta.ndim - 1)))
        return delta
