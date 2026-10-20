"""Refiner training objective.

Writing the DiT's prediction in x-space as ``x_hat1`` (``Transport.to_x_pred``) and
the flow-matching target as ``v_t = (x_t - x1)/t_s``,

    r = v_t - v_hat = (x_hat1 - x1) / t_s

which is exactly the reference's ``velocity_residual = training_target - noise_pred``
with ``sigma == t``. ``MSE(g, r)`` is therefore identical to ``MSE(v_hat + g, v_t)``.
"""

from __future__ import annotations

from typing import Dict

import torch


def compute_refiner_terms(
    refiner,
    capture,
    *,
    x_pred: torch.Tensor,
    x1: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    t_safe: torch.Tensor,
    config,
) -> Dict[str, torch.Tensor]:
    """Returns ``loss_refiner`` plus detached monitors (rel / cos / band fraction)."""
    payload = capture.finalize()

    # clean_prediction and the target are always detached (reference semantics);
    # only the captured hidden states / time embedding may keep the DiT graph,
    # which the capture object controls via `config.grad_to_dit`.
    x_pred = x_pred.detach()
    target = (x_pred - x1.detach()) / t_safe
    refiner_input = (xt.detach() - x_pred) / t_safe if config.input_v_prediction else x_pred

    pred = refiner(refiner_input, payload["t_emb"], payload["cond_emb"], payload["hidden"])

    # Reduce with vector_norm / einsum rather than materialising squared copies:
    # at micro-batch 128 every full-size fp32 tensor here is 134 MB, and the naive
    # `(a - b).pow(2).mean(1)` form kept five of them alive at once.
    pred_f = pred.float().flatten(1)
    target_f = target.float().flatten(1)
    dim = pred_f.shape[1]

    diff = pred_f - target_f
    mse = torch.linalg.vector_norm(diff, dim=1).square() / dim
    del diff
    pred_norm = torch.linalg.vector_norm(pred_f, dim=1)
    target_norm = torch.linalg.vector_norm(target_f, dim=1)
    baseline = target_norm.square() / dim

    per_sample = mse / (baseline.detach() + 1e-8) if config.loss_normalize else mse
    cos = torch.einsum("bd,bd->b", pred_f, target_f) / (pred_norm * target_norm + 1e-8)
    if config.cosine_lambda > 0:
        # cos(0, r) is undefined, and the eps in the denominator hands the zero
        # point a ~1/eps gradient: measured 1.1e7 against an MSE gradient of
        # 5.3e-4, so after clip_grad_norm_ the MSE term is 1e-10 relative and is
        # simply rounded away in fp32. With a zero-init head that is *every*
        # sample on step 1. Freeze the term at its cos=0 value there (identical
        # loss value, no gradient) and let MSE move the output off zero first --
        # same guard as the reference's `has_direction` branch.
        has_direction = (pred_norm > 1e-8) & (target_norm > 1e-8)
        direction = torch.where(has_direction, 1.0 - cos, torch.ones_like(cos))
        per_sample = per_sample + config.cosine_lambda * direction

    # Multiply by a 0/1 mask instead of selecting rows: every refiner parameter
    # must receive a gradient on every step for DDP(find_unused_parameters=False).
    mask = ((t >= config.t_min) & (t <= config.t_max)).to(per_sample.dtype)
    denom = mask.sum().clamp_min(1.0)

    return {
        "loss_refiner": (per_sample * mask).sum() / denom,
        "refiner_rel": ((mse * mask).sum() / ((baseline * mask).sum() + 1e-8)).detach(),
        "refiner_cos": ((cos * mask).sum() / denom).detach(),
        "refiner_band_frac": (mask.mean()).detach(),
    }
