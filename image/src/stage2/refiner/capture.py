"""Forward-hook capture of the stage-2 DiT's internals for the refiner.

Mirrors ``corrflow``'s ``WanHiddenStateCapture``: grabs the per-sample time
embedding, the condition tokens and a handful of transformer-block hidden states
without touching the DiT definition. Hooks can be attached to several DiTs (live
+ EMA); only one of them is ever mid-forward, so the last write is the live one.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch


class DiTHiddenCapture:
    """Collects ``t_emb`` / ``cond_emb`` / selected block hidden states."""

    def __init__(self, layers: Sequence[int], detach: bool = True):
        # 1-indexed encoder-block depths, matching the reference's "3,11,19,29".
        self.layers = tuple(int(l) for l in layers)
        self.detach = bool(detach)
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._num_patches = None
        self.t_emb = None
        self.cond_emb = None
        self.hidden: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ setup
    def attach(self, model) -> "DiTHiddenCapture":
        """Register hooks on one DiT. Safe to call for several models."""
        num_enc_blocks = model.num_enc_blocks
        for layer in self.layers:
            if not 1 <= layer <= num_enc_blocks:
                raise ValueError(
                    f"refiner selected layer {layer} outside the encoder trunk "
                    f"(1..{num_enc_blocks}); decoder blocks have a different width."
                )
        self._num_patches = model.s_embedder.num_patches

        def t_hook(_module, _inp, output):
            emb = output[0] if isinstance(output, tuple) else output
            self.t_emb = emb.detach() if self.detach else emb

        def ctx_hook(_module, _inp, output):
            # Condition tokens are always detached (as in the reference).
            self.cond_emb = output.detach()

        self._handles.append(model.t_embedder.register_forward_hook(t_hook))
        self._handles.append(model.ctx_embedder.register_forward_hook(ctx_hook))
        for layer in self.layers:
            self._handles.append(
                model.blocks[layer - 1].register_forward_hook(self._make_block_hook(layer))
            )
        return self

    def _make_block_hook(self, layer: int):
        def hook(_module, _inp, output):
            h = output[:, : self._num_patches, :]
            self.hidden[layer] = h.detach() if self.detach else h
        return hook

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -------------------------------------------------------------- consumption
    def clear(self) -> None:
        self.t_emb = None
        self.cond_emb = None
        self.hidden = {}

    def finalize(self) -> Dict[str, torch.Tensor]:
        """Returns ``t_emb (B,1,D)``, ``cond_emb (B,Nc,D)``, ``hidden (B,L,N,D)``."""
        if self.t_emb is None or self.cond_emb is None or len(self.hidden) != len(self.layers):
            raise RuntimeError(
                "DiTHiddenCapture is empty; the DiT forward must run before the refiner "
                f"(t_emb={self.t_emb is not None}, cond_emb={self.cond_emb is not None}, "
                f"hidden={sorted(self.hidden)} vs layers={list(self.layers)})."
            )
        hidden = torch.stack([self.hidden[l] for l in self.layers], dim=1)
        return {"t_emb": self.t_emb, "cond_emb": self.cond_emb, "hidden": hidden}

    @staticmethod
    def slice_payload(payload: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
        """Take the first ``n`` rows (the conditional branch under CFG)."""
        out = {}
        for key, value in payload.items():
            if value.shape[0] == n:
                out[key] = value
            elif value.shape[0] % n == 0:
                out[key] = value[:n]
            else:
                raise AssertionError(
                    f"captured '{key}' batch {value.shape[0]} is not a multiple of {n}"
                )
        return out
