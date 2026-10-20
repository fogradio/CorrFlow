"""Stage 2 shared utilities.

Contains config validation + helpers shared between stage2/engine.py
"""

from __future__ import annotations

import dataclasses
from copy import deepcopy

import torch
from torch.cuda.amp import autocast

from configs.stage2 import Stage2Config
from stage2.models.embedders import TextEncoder
from utils.dist_utils import main_process_first


def validate_stage2_config(config: Stage2Config) -> None:
    """Validate a Stage2Config for consistency."""
    if not config.stage_1.target:
        raise ValueError("Config must provide stage_1.target (RAE model).")
    if not config.stage_2.target:
        raise ValueError("Config must provide stage_2.target (DiT model).")

    # REPA validation
    repa = config.repa
    if repa.use_repa:
        if not repa.target_encoder:
            raise ValueError("repa.target_encoder is required when use_repa=True.")

    # Gradient accumulation
    if config.training.grad_accum_steps < 1:
        raise ValueError("training.grad_accum_steps must be >= 1.")

    # Conditioning
    cond = config.conditioning
    if cond.type == "text" and cond.text_encoder is None:
        raise ValueError("conditioning.text_encoder must be set when conditioning.type='text'.")

    # Self-Flow deliberately supports only the ImageNet label-conditioned path
    # in this implementation. Keeping incompatible auxiliary objectives out of
    # the same run also makes the two-view loss and its accounting unambiguous.
    sf = config.self_flow
    if sf.enabled:
        if not 0.0 <= sf.mask_ratio <= 0.5:
            raise ValueError("self_flow.mask_ratio must satisfy 0 <= mask_ratio <= 0.5.")
        if sf.representation_loss_weight < 0.0:
            raise ValueError("self_flow.representation_loss_weight must be non-negative.")
        if sf.expected_num_parameters <= 0:
            raise ValueError("self_flow.expected_num_parameters must be positive.")
        if sf.parameter_tolerance < 0.0:
            raise ValueError("self_flow.parameter_tolerance must be non-negative.")

        depth = config.stage_2.params.get("depth")
        if not isinstance(depth, int):
            raise ValueError("Self-Flow requires stage_2.params.depth to be a scalar integer.")
        if not 1 <= sf.student_layer < sf.teacher_layer <= depth:
            raise ValueError(
                "Self-Flow layers must satisfy "
                f"1 <= student_layer < teacher_layer <= depth; got "
                f"{sf.student_layer}, {sf.teacher_layer}, depth={depth}."
            )
        if config.conditioning.type != "label":
            raise ValueError("Self-Flow currently supports only label conditioning.")
        if config.transport.prediction != "velocity":
            raise ValueError("Self-Flow requires transport.prediction='velocity'.")
        if config.repa.use_repa:
            raise ValueError("Self-Flow cannot be combined with REPA.")
        if config.internal_guidance.base_model_depth is not None or config.guidance.use_ig:
            raise ValueError("Self-Flow cannot be combined with internal guidance (IG).")
        if config.refiner is not None:
            raise ValueError("Self-Flow cannot be combined with refiner training.")
    # Refiner
    cp = config.refiner
    if cp is not None:
        if not cp.model.target:
            raise ValueError("refiner.model.target is required when the refiner section is present.")
        if not cp.selected_layers:
            raise ValueError("refiner.selected_layers must be a non-empty list of encoder depths.")
        depth = config.stage_2.params.get("depth")
        num_enc_blocks = depth[0] if isinstance(depth, (list, tuple)) else depth
        if num_enc_blocks is not None:
            for layer in cp.selected_layers:
                if not 1 <= layer <= num_enc_blocks:
                    raise ValueError(
                        f"refiner.selected_layers entry {layer} is outside the encoder trunk "
                        f"(1..{num_enc_blocks}); decoder blocks have a different width."
                    )
        if (cp.t_min, cp.t_max) != (0.0, 1.0):
            raise ValueError("Refiner training requires the full [0, 1] interval.")
        if (cp.sampling.t_min, cp.sampling.t_max) != (0.0, 1.0):
            raise ValueError("Refiner sampling requires the full [0, 1] interval.")
        if cp.grad_to_dit and cp.loss_weight == 0:
            raise ValueError("refiner.grad_to_dit=True with loss_weight=0 has no effect.")


##############################################################
# Shared helpers used by both stage2/engine
##############################################################
def apply_cfg_dropout(model_conds, model_conds_null, cfg_dropout_prob=0.1):
    if isinstance(model_conds['context'], dict):
        from stage2 import nwm_cond
        return nwm_cond.apply_cfg_dropout(model_conds, model_conds_null, cfg_dropout_prob)
    mask = torch.rand(model_conds['context'].shape[0], device=model_conds['context'].device) < cfg_dropout_prob
    return {
        k: torch.where(mask.view(-1, *([1]*(v.ndim-1))), model_conds_null[k], v) if v is not None else None
        for k, v in model_conds.items()
    }, mask


def get_null_cond(text_encoder,conditioning_type, num_classes, batch_size, device):
    if conditioning_type == "text":
        _null_context, _null_attn_mask = encode_text(text_encoder, [""])
    else:
        _null_context, _null_attn_mask = torch.tensor([num_classes], device=device), None
    rtn = dict(context=_null_context, attn_mask=_null_attn_mask)
    rtn = {k: v.expand(batch_size, *v.shape[1:]) if v is not None else None for k, v in rtn.items()}
    return rtn


def setup_text_encoder(config, rank, device):
    """Build text encoder if conditioning.type == 'text', else return None.

    Side effect: sets config.conditioning.context_dim from the encoder's feature_dim.
    """
    if config.conditioning.type != "text":
        return None
    with main_process_first(rank):
        text_encoder = TextEncoder(**dataclasses.asdict(config.conditioning.text_encoder)).to(device)
    config.conditioning.context_dim = text_encoder.feature_dim
    return text_encoder


def encode_text(text_encoder, y):
    """Encode text conditions. Returns (encoder_hidden_states, encoder_attention_mask)."""
    with torch.no_grad():
        enc_out = text_encoder(y)
        return enc_out["tokens"], enc_out["attention_mask"]


def get_fixed_viz_batch_conditions(viz_fixed, y, condition_type, text_encoder, device):
    """Get fixed conditions for the first batch for consistent visualization."""
    if viz_fixed['context'] is not None:
        return viz_fixed
    n = viz_fixed['zs'].shape[0]
    if condition_type == "label":
        viz_fixed['context'] = y[:n].clone().to(device)
    else:
        with torch.no_grad():
            enc_out = text_encoder(y[:n])
            viz_fixed['context'] = enc_out["tokens"]
            viz_fixed['attn_mask'] = enc_out["attention_mask"]
    return viz_fixed


def sample_and_decode(
    zs, context, attn_mask,
    eval_sampler, model_fn, sample_model_kwargs, rae,
    use_guidance, condition_type, text_encoder, num_classes, device, autocast_kwargs,
):
    """Generate and decode samples, handling guidance doubling."""
    n = zs.shape[0]
    if use_guidance:
        zs = torch.cat([zs, zs], dim=0)
        if isinstance(context, dict):
            null = {k: torch.zeros_like(v) for k, v in context.items()}
            context = {k: torch.cat([context[k], null[k]], dim=0) for k in context}
            attn_mask = None
        else:
            if condition_type == "text":
                context_null, attn_mask_null = encode_text(text_encoder, [""] * n)
            else:
                context_null = torch.full((n,), num_classes, device=device)
                attn_mask_null = None
            context = torch.cat([context, context_null], dim=0)
            if attn_mask is not None and attn_mask_null is not None:
                attn_mask = torch.cat([attn_mask, attn_mask_null], dim=0)

    kwargs = deepcopy(sample_model_kwargs)
    kwargs.update(context=context, attn_mask=attn_mask)
    with autocast(**autocast_kwargs):
        samples = eval_sampler(zs, model_fn, **kwargs)[-1]
        if use_guidance:
            samples = samples.chunk(2, dim=0)[0]
    return rae.decode(samples).cpu().float()
