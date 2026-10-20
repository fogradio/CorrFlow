"""Refiner for RAEv2 stage-2.

Uses a Wan-style AdaLN trunk with
interleaved self-/cross-attention blocks and an ``h_fuse`` memory built from the
backbone's selected hidden states. Three things change for the image setting:

* 3-D RoPE -> the stage-2 2-D image RoPE (``stage2.models.model_utils.RoPE``),
* T5 context -> the DiT's own ``ctx_embedder`` output as ``cond_emb``,
* Wan's global time vector -> the DiT's ``t_emb_base``.

Output semantics: the velocity-space residual ``r = v_t - v_hat``; see
``stage2.refiner.losses``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2.models.model_utils import RMSNorm, RoPE


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, rope):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
        q, k = rope(q), rope(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.permute(0, 2, 1, 3).reshape(B, N, D))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, 2 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, memory):
        B, N, D = x.shape
        Nk = memory.shape[1]
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(memory).view(B, Nk, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k, v = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.permute(0, 2, 1, 3).reshape(B, N, D))


class FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        hidden = int(round(dim * mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.self_attn = SelfAttention(dim, num_heads)
        self.norm_ff = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ffn = FFN(dim, mlp_ratio)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, memory, rope, t_mod):
        mods = (self.modulation.to(dtype=t_mod.dtype) + t_mod).chunk(6, dim=1)
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = (m.squeeze(1) for m in mods)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            _modulate(self.norm_sa(x), shift_sa, scale_sa), rope)
        x = x + gate_ff.unsqueeze(1) * self.ffn(
            _modulate(self.norm_ff(x), shift_ff, scale_ff))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm_ca = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.norm_ff = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ffn = FFN(dim, mlp_ratio)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, memory, rope, t_mod):
        mods = (self.modulation.to(dtype=t_mod.dtype) + t_mod).chunk(6, dim=1)
        shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff = (m.squeeze(1) for m in mods)
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(
            _modulate(self.norm_ca(x), shift_ca, scale_ca), memory)
        x = x + gate_ff.unsqueeze(1) * self.ffn(
            _modulate(self.norm_ff(x), shift_ff, scale_ff))
        return x


class Refiner(nn.Module):
    """Residual-predicting side head for the stage-2 DiT."""

    def __init__(
        self,
        in_channels: int = 1024,
        input_size: int = 16,
        patch_size: int = 1,
        dim: int = 768,
        depth: int = 10,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        time_emb_dim: int = 1440,
        cond_emb_dim: int = 1440,
        selected_hidden_dim: int = 1440,
        n_selected_layers: int = 4,
        zero_init_output: bool = True,
        normalize_selected_hidden: bool = False,
        inject_selected_hidden: bool = False,
        drop_condition: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        assert input_size % patch_size == 0
        self.dim = dim
        self.depth = depth
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.grid_size = input_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.n_selected_layers = n_selected_layers
        self.selected_hidden_dim = selected_hidden_dim
        self.drop_condition = bool(drop_condition)

        self.patch_embed = nn.Linear(self.patch_dim, dim)
        self.hidden_inject = (
            nn.Linear(n_selected_layers * selected_hidden_dim, dim)
            if inject_selected_hidden else None
        )

        self.rope = RoPE(dim // num_heads, self.num_patches, 0, theta=rope_theta)

        self.time_embedding = nn.Sequential(
            nn.Linear(time_emb_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6, bias=True),
        )

        self.cond_proj = nn.Linear(cond_emb_dim, dim)
        self.shs_norm = RMSNorm(selected_hidden_dim) if normalize_selected_hidden else None
        self.h_fuse_mlp = nn.Sequential(
            nn.Linear(n_selected_layers * selected_hidden_dim, dim, bias=True),
            nn.GELU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.seg_embed = nn.Parameter(torch.zeros(2, dim))
        nn.init.trunc_normal_(self.seg_embed, std=0.02)
        self.memory_norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)

        self.blocks = nn.ModuleList([
            SelfAttnBlock(dim, num_heads, mlp_ratio) if i % 2 == 0
            else CrossAttnBlock(dim, num_heads, mlp_ratio)
            for i in range(depth)
        ])

        self.final_norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.final_modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)
        self.head = nn.Linear(dim, self.patch_dim, bias=True)
        if zero_init_output:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        if self.drop_condition:
            # cond_proj is unreachable; freeze so DDP(find_unused_parameters=False)
            # never sees a parameter without a gradient.
            self.cond_proj.requires_grad_(False)

    def _patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)
        return x

    def _unpatchify(self, tokens):
        B = tokens.shape[0]
        p, g, c = self.patch_size, self.grid_size, self.in_channels
        x = tokens.reshape(B, g, g, c, p, p).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, c, g * p, g * p)

    def _encode_memory(self, cond_emb, selected_hidden_states):
        B, L, N, D_in = selected_hidden_states.shape
        assert L == self.n_selected_layers, f"expected {self.n_selected_layers} layers, got {L}"
        assert D_in == self.selected_hidden_dim, f"expected hidden_dim={self.selected_hidden_dim}, got {D_in}"
        if self.shs_norm is not None:
            selected_hidden_states = self.shs_norm(selected_hidden_states)
        h_cat = selected_hidden_states.permute(0, 2, 1, 3).reshape(B, N, L * D_in)
        h_fuse = self.h_fuse_mlp(h_cat)
        if self.drop_condition:
            return self.memory_norm(h_fuse + self.seg_embed[1])
        c_ = self.cond_proj(cond_emb) + self.seg_embed[0]
        h_ = h_fuse + self.seg_embed[1]
        return self.memory_norm(torch.cat([c_, h_], dim=1))

    def forward(self, x, t_emb, cond_emb, selected_hidden_states):
        """x: (B, C, H, W); t_emb: (B, Dt) or (B, 1, Dt); cond_emb: (B, Nc, Dc);
        selected_hidden_states: (B, L, N, Dh). Returns (B, C, H, W)."""
        tokens = self.patch_embed(self._patchify(x))

        if self.hidden_inject is not None:
            B_, L_, N_, D_ = selected_hidden_states.shape
            assert N_ == tokens.shape[1], (
                f"hidden tokens ({N_}) misaligned with trunk tokens ({tokens.shape[1]})")
            h_tok = selected_hidden_states.permute(0, 2, 1, 3).reshape(B_, N_, L_ * D_)
            tokens = tokens + self.hidden_inject(h_tok.to(tokens.dtype))

        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        t = self.time_embedding(t_emb.to(tokens.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        memory = self._encode_memory(cond_emb.to(tokens.dtype), selected_hidden_states.to(tokens.dtype))

        for blk in self.blocks:
            tokens = blk(tokens, memory, self.rope, t_mod)

        shift, scale = (self.final_modulation.to(dtype=t.dtype) + t.unsqueeze(1)).chunk(2, dim=1)
        tokens = self.final_norm(tokens) * (1 + scale) + shift
        return self._unpatchify(self.head(tokens))
