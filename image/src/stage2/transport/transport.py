import math
from dataclasses import dataclass

import torch as th
import torch.nn.functional as F

from stage2.utils import apply_cfg_dropout


def _expand_t(t, x):
    """Expand scalar or patch-token timesteps to an image-like latent."""
    if t.ndim == 1:
        return t.view(t.size(0), *([1] * (len(x.size()) - 1)))
    if t.ndim != 2 or x.ndim != 4:
        raise ValueError(
            f"Expected timesteps [B] or [B,N] for latent [B,C,H,W], got "
            f"{tuple(t.shape)} and {tuple(x.shape)}."
        )

    num_tokens = t.shape[1]
    token_side = math.isqrt(num_tokens)
    if token_side * token_side != num_tokens:
        raise ValueError(f"Per-token timesteps require a square grid, got N={num_tokens}.")
    height, width = x.shape[-2:]
    if height % token_side != 0 or width % token_side != 0:
        raise ValueError(
            f"Token grid {token_side}x{token_side} does not tile latent {height}x{width}."
        )
    expanded = t.reshape(t.shape[0], 1, token_side, token_side)
    expanded = expanded.repeat_interleave(height // token_side, dim=2)
    return expanded.repeat_interleave(width // token_side, dim=3)


@dataclass
class DualTimestepBatch:
    """Two correlated Self-Flow views built from one data/noise pair."""

    student_input: th.Tensor
    student_timesteps: th.Tensor
    teacher_input: th.Tensor
    teacher_timestep: th.Tensor
    velocity_target: th.Tensor
    mask: th.Tensor


def build_dual_timestep_batch(
    data: th.Tensor,
    noise: th.Tensor,
    t: th.Tensor,
    s: th.Tensor,
    mask: th.Tensor,
) -> DualTimestepBatch:
    """Construct mixed student and uniformly-cleaner teacher views.

    ``t`` and ``s`` are independent per-sample draws. The same Gaussian noise
    is reused in both views, exactly as required by the Self-Flow objective.
    """
    if data.shape != noise.shape:
        raise ValueError("data and noise must have identical shapes.")
    if t.shape != s.shape or t.ndim != 1 or t.shape[0] != data.shape[0]:
        raise ValueError("t and s must both have shape [B].")
    if mask.ndim != 2 or mask.shape[0] != data.shape[0]:
        raise ValueError("mask must have shape [B,N].")

    mask = mask.to(device=data.device, dtype=th.bool)
    student_timesteps = th.where(mask, s.unsqueeze(1), t.unsqueeze(1))
    teacher_timestep = th.minimum(t, s)
    student_time_grid = _expand_t(student_timesteps, data)
    teacher_time_grid = _expand_t(teacher_timestep, data)
    student_input = (1 - student_time_grid) * data + student_time_grid * noise
    teacher_input = (1 - teacher_time_grid) * data + teacher_time_grid * noise
    return DualTimestepBatch(
        student_input=student_input,
        student_timesteps=student_timesteps,
        teacher_input=teacher_input,
        teacher_timestep=teacher_timestep,
        velocity_target=noise - data,
        mask=mask,
    )


def cosine_representation_loss(student_features, teacher_features):
    """Paper Eq. (6): negative mean token-wise cosine similarity."""
    return -F.cosine_similarity(
        student_features.float(),
        teacher_features.float(),
        dim=-1,
    ).mean()


def get_time_sampler(time_dist_type: str):
    parts = time_dist_type.split("_")
    name = parts[0]
    if name == "logit-normal":
        assert len(parts) == 3, f"Expected 'logit-normal_MU_SIGMA', got '{time_dist_type}'"
        mu, sigma = float(parts[1]), float(parts[2])
        assert sigma > 0, "sigma must be > 0"
        return lambda bs: (th.randn(bs) * sigma + mu).sigmoid()
    else:
        raise NotImplementedError(f"Unknown time distribution: {time_dist_type}")


class Transport:
    def __init__(self, prediction="velocity", time_dist_type="logit-normal_0_1", time_dist_shift=1.0, t_eps=0.05,
                 self_flow_config=None):
        self.prediction = prediction
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        self.t_eps = t_eps
        self.time_sampler = get_time_sampler(time_dist_type)
        self.self_flow_config = self_flow_config
        # Optional sampling-time refiner correction; see stage2.refiner.runner.
        self.refiner_runner = None

    def sample(self, x1):
        x0 = th.randn_like(x1)
        t = self.time_sampler(x1.shape[0]).to(x1)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    def _sample_timestep(self, batch_size, reference):
        timestep = self.time_sampler(batch_size).to(reference)
        return self.time_dist_shift * timestep / (
            1 + (self.time_dist_shift - 1) * timestep
        )

    def sample_dual_timestep_batch(self, data, num_tokens):
        t = self._sample_timestep(data.shape[0], data)
        s = self._sample_timestep(data.shape[0], data)
        noise = th.randn_like(data)
        mask = th.rand(data.shape[0], num_tokens, device=data.device) < self.self_flow_config.mask_ratio
        return build_dual_timestep_batch(data, noise, t, s, mask)

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################
    def training_losses(self, model, x1, model_kwargs={}, model_kwargs_null={}, z_clean=None, repa_coeff=None, base_model_coeff=1.0, cfg_dropout_prob=0.1,
                        refiner=None, refiner_capture=None, refiner_config=None,
                        teacher_model=None):
        model_kwargs, _ = apply_cfg_dropout(model_kwargs, model_kwargs_null, cfg_dropout_prob)

        if self.self_flow_config is not None and self.self_flow_config.enabled:
            return self._self_flow_training_losses(
                model,
                teacher_model,
                x1,
                model_kwargs,
            )

        t, x0, x1 = self.sample(x1)
        xt = (1 - _expand_t(t, x1)) * x1 + _expand_t(t, x1) * x0
        t_safe = _expand_t(t, xt).clamp_min(self.t_eps)
        vt = (xt - x1) / t_safe

        enable_repa = z_clean is not None and repa_coeff is not None
        zt_pred = None
        if enable_repa:
            model_output, zt_pred = model(xt, t, return_intermediate=True, **model_kwargs)
        else:
            model_output = model(xt, t, **model_kwargs)

        # Handle Internal Guidance dual-output models (full, base)
        base_output = None
        if isinstance(model_output, tuple) and len(model_output) == 2:
            model_output, base_output = model_output

        terms = {'loss': self.compute_loss(model_output, vt, xt, t)}
        if base_output is not None:
            loss_base = self.compute_loss(base_output, vt, xt, t)
            terms['loss'] = terms['loss'] + base_model_coeff * loss_base
            terms['loss_base'] = loss_base
        if enable_repa and zt_pred is not None:
            terms['loss_repa'] = repa_coeff * F.mse_loss(zt_pred, z_clean)

        if refiner is not None:
            from stage2.refiner import compute_refiner_terms
            terms.update(compute_refiner_terms(
                refiner, refiner_capture,
                x_pred=self.to_x_pred(model_output, xt, t), x1=x1, xt=xt, t=t, t_safe=t_safe,
                config=refiner_config,
            ))

        return terms

    def _self_flow_training_losses(self, model, teacher_model, data, model_kwargs):
        if teacher_model is None:
            raise ValueError("Self-Flow training requires an EMA teacher model.")

        student_module = model.module if hasattr(model, "module") else model
        num_tokens = getattr(student_module, "num_patches", None)
        if num_tokens is None:
            raise AttributeError("Self-Flow model must expose num_patches.")
        batch = self.sample_dual_timestep_batch(data, num_tokens)

        model_output, student_features = model(
            batch.student_input,
            batch.student_timesteps,
            return_features=self.self_flow_config.student_layer,
            **model_kwargs,
        )
        with th.no_grad():
            _, teacher_features = teacher_model(
                batch.teacher_input,
                batch.teacher_timestep,
                return_raw_features=self.self_flow_config.teacher_layer,
                **model_kwargs,
            )

        generation_elementwise = self.compute_loss(
            model_output,
            batch.velocity_target,
            batch.student_input,
            batch.student_timesteps,
        )
        generation_loss = generation_elementwise.mean()
        representation_loss = cosine_representation_loss(student_features, teacher_features)
        total_loss = generation_loss + (
            self.self_flow_config.representation_loss_weight * representation_loss
        )
        return {
            # Keep the legacy elementwise key for callers that inspect it.
            "loss": generation_elementwise,
            "loss_generation": generation_loss,
            "loss_representation": representation_loss,
            "loss_total": total_loss,
            "self_flow_mask_ratio": batch.mask.float().mean(),
        }

    def to_x_pred(self, output, xt, t):
        """Model output expressed in x-space (the predicted clean latent)."""
        if self.prediction == "x":
            return output
        return xt - _expand_t(t, xt).clamp_min(self.t_eps) * output

    def convert_model_pred(self, output, xt, t):
        # Unify model output to v-pred
        if self.prediction == "velocity":
            return output
        elif self.prediction == "x":
            t_safe = _expand_t(t, xt).clamp_min(self.t_eps)
            return (xt - output) / t_safe
        raise ValueError(f"Unknown prediction type: {self.prediction}")

    def compute_loss(self, output, vt, xt, t):
        output = self.convert_model_pred(output, xt, t)
        return (output - vt) ** 2

    def get_drift(self):
        def body_fn(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            velocity = self.convert_model_pred(model_output, x, t)
            # Correct after CFG/IG have been combined, so the correction is not
            # scaled by cfg_scale.
            if self.refiner_runner is not None:
                velocity = velocity + self.refiner_runner(
                    x, t, self.to_x_pred(model_output, x, t), velocity)
            return velocity
        return body_fn
