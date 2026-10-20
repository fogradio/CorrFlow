"""Self-Flow DiT with per-token adaptive LayerNorm conditioning.

The architecture follows the official Self-Flow ImageNet implementation while
accepting RAE's native ``[B, C, H, W]`` latents and returning a velocity tensor
of the same shape. Scalar timesteps are used during ODE sampling; ``[B, N]``
timesteps enable Dual-Timestep Scheduling during training.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp


def _modulate_per_token(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return x * (1 + scale) + shift


def _get_1d_sincos_pos_embed(
    embed_dim: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"1D positional embedding width must be even, got {embed_dim}.")
    omega = torch.arange(embed_dim // 2, device=positions.device, dtype=torch.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    angles = positions.reshape(-1).to(torch.float64).unsqueeze(1) * omega.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=1).float()


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return fixed row/column sin-cos embeddings with shape ``[N, D]``."""
    if embed_dim % 4 != 0:
        raise ValueError(f"2D positional embedding width must be divisible by 4, got {embed_dim}.")
    coords = torch.arange(grid_size, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    emb_y = _get_1d_sincos_pos_embed(embed_dim // 2, grid_y)
    emb_x = _get_1d_sincos_pos_embed(embed_dim // 2, grid_x)
    return torch.cat([emb_y, emb_x], dim=1)


class PatchEmbed(nn.Module):
    """Patchify an image-like latent and linearly project every patch."""

    def __init__(self, input_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError("input_size must be divisible by patch_size.")
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = input_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected latent [B,C,H,W], got shape {tuple(x.shape)}.")
        b, c, h, w = x.shape
        p = self.patch_size
        if h != self.input_size or w != self.input_size:
            raise ValueError(
                f"Expected {self.input_size}x{self.input_size} latent, got {h}x{w}."
            )
        patches = x.reshape(b, c, h // p, p, w // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(b, self.num_patches, c * p * p)
        return self.proj(patches)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep features followed by a two-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([args.cos(), args.sin()], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """ImageNet label embedding with one explicit null class for CFG."""

    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels.reshape(-1))


class PerTokenDiTBlock(nn.Module):
    """DiT block whose adaLN parameters vary independently for every token."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_hidden_size: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_size,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        modulation = self.adaLN_modulation(condition)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            modulation.chunk(6, dim=-1)
        )
        x = x + gate_attn * self.attn(
            _modulate_per_token(self.norm1(x), shift_attn, scale_attn)
        )
        x = x + gate_mlp * self.mlp(
            _modulate_per_token(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class PerTokenFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size**2 * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(_modulate_per_token(self.norm_final(x), shift, scale))


class SimpleHead(nn.Module):
    """Official two-layer Self-Flow projection head."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim + out_dim)
        self.linear2 = nn.Linear(in_dim + out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class SelfFlowDiT(nn.Module):
    """Label-conditioned Self-Flow DiT for RAE latent-space generation."""

    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 1024,
        hidden_size: int = 1280,
        depth: int = 30,
        num_heads: int = 20,
        mlp_ratio: float = 4.0,
        mlp_hidden_size: Optional[int] = None,
        num_classes: int = 1000,
        condition_type: str = "label",
        context_dim: Optional[int] = None,
        cond_arch=None,
        use_cfg_conds: bool = False,
    ) -> None:
        super().__init__()
        del context_dim, cond_arch
        if condition_type != "label":
            raise ValueError("SelfFlowDiT supports only label conditioning.")
        if use_cfg_conds:
            raise ValueError("SelfFlowDiT does not support learned CFG-scale tokens.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_hidden_size = mlp_hidden_size or int(hidden_size * mlp_ratio)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.num_patches = self.x_embedder.num_patches
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        pos_embed = get_2d_sincos_pos_embed(
            hidden_size,
            self.x_embedder.grid_size,
            device=self.x_embedder.proj.weight.device,
        )
        # The official implementation counts this fixed tensor as a parameter.
        self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)

        self.blocks = nn.ModuleList([
            PerTokenDiTBlock(hidden_size, num_heads, self.mlp_hidden_size)
            for _ in range(depth)
        ])
        self.final_layer = PerTokenFinalLayer(hidden_size, patch_size, in_channels)
        self.projector = SimpleHead(hidden_size, hidden_size)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Meta-device construction is used solely for exact parameter counting.
        if self.x_embedder.proj.weight.is_meta:
            return

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, _ = tokens.shape
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(f"Expected a square token grid, got {n} tokens.")
        p, c = self.patch_size, self.out_channels
        x = tokens.reshape(b, side, side, c, p, p)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, side * p, side * p)

    def _embed_timesteps(self, timesteps: torch.Tensor, batch_size: int) -> torch.Tensor:
        if timesteps.ndim == 1:
            if timesteps.shape[0] != batch_size:
                raise ValueError("Scalar timestep batch does not match latent batch size.")
            return self.t_embedder(timesteps).unsqueeze(1).expand(-1, self.num_patches, -1)
        if timesteps.ndim == 2:
            if timesteps.shape != (batch_size, self.num_patches):
                raise ValueError(
                    f"Per-token timesteps must have shape {(batch_size, self.num_patches)}, "
                    f"got {tuple(timesteps.shape)}."
                )
            embedded = self.t_embedder(timesteps.reshape(-1))
            return embedded.reshape(batch_size, self.num_patches, self.hidden_size)
        raise ValueError(
            f"Timesteps must have shape [B] or [B,N], got {tuple(timesteps.shape)}."
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        return_features: int | bool = False,
        return_raw_features: int | bool = False,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **condition_kwargs,
    ):
        del attn_mask, condition_kwargs
        if return_features and return_raw_features:
            raise ValueError("Only one feature return mode can be selected.")
        requested_layer = int(return_features or return_raw_features or 0)
        if requested_layer and not 1 <= requested_layer <= self.depth:
            raise ValueError(
                f"Requested feature layer {requested_layer} outside model depth 1..{self.depth}."
            )

        tokens = self.x_embedder(x)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype)
        t_embedding = self._embed_timesteps(t, tokens.shape[0])
        y_embedding = self.y_embedder(context).unsqueeze(1).expand(-1, self.num_patches, -1)
        condition = t_embedding + y_embedding

        features = None
        for index, block in enumerate(self.blocks, start=1):
            tokens = block(tokens, condition)
            if index == requested_layer:
                features = self.projector(tokens) if return_features else tokens

        output = self._unpatchify(self.final_layer(tokens, condition))
        if requested_layer:
            return output, features
        return output


__all__ = ["SelfFlowDiT", "SimpleHead", "get_2d_sincos_pos_embed"]
