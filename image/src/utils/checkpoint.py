"""Checkpoint save/load utilities for Stage 1 and Stage 2 training."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def save_stage1_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> None:
    """Save Stage 1 training checkpoint (model + discriminator)."""
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_stage1_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 1 training checkpoint. Returns (epoch, step)."""
    checkpoint = torch.load(path, map_location="cpu")
    if "model" not in checkpoint or "ema" not in checkpoint:
        raise KeyError(
            f"{path} is not a Stage-2 training checkpoint (keys: {sorted(checkpoint)[:8]}). "
            "Anything matching 'ep-<digits>' in the checkpoint dir is treated as a resume "
            "candidate; move stray weight dumps out of it."
        )
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    disc.load_state_dict(checkpoint["disc"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def save_stage2_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    refiner: Optional[torch.nn.Module] = None,
    refiner_ema: Optional[torch.nn.Module] = None,
    refiner_optimizer: Optional[torch.optim.Optimizer] = None,
    refiner_scheduler: Optional[LambdaLR] = None,
) -> None:
    """Save Stage 2 training checkpoint (optionally with the refiner side head)."""
    state = {
        "step": step,
        "epoch": epoch,
        "model": _unwrap(model).state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    if refiner is not None:
        state["refiner"] = _unwrap(refiner).state_dict()
        state["refiner_ema"] = refiner_ema.state_dict() if refiner_ema is not None else None
        state["refiner_optimizer"] = (
            refiner_optimizer.state_dict() if refiner_optimizer is not None else None
        )
        state["refiner_scheduler"] = (
            refiner_scheduler.state_dict() if refiner_scheduler is not None else None
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def save_refiner_weights(path: str, refiner: torch.nn.Module) -> None:
    """Dump a flat refiner state dict (mirrors the reference's refiner_step-N.pt).

    Keep these out of the checkpoint dir itself: find_resume_checkpoint() globs
    ``*.pt`` there and ranks candidates by an ``ep-(\d+)`` regex, which also
    matches ``refiner_ep-0000000.pt`` -- it would then try to resume from a flat
    refiner state dict and die with ``KeyError: 'model'``.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(_unwrap(refiner).state_dict(), path)


def load_stage2_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    refiner: Optional[torch.nn.Module] = None,
    refiner_ema: Optional[torch.nn.Module] = None,
    refiner_optimizer: Optional[torch.optim.Optimizer] = None,
    refiner_scheduler: Optional[LambdaLR] = None,
) -> Tuple[int, int]:
    """Load Stage 2 training checkpoint. Returns (epoch, step).

    A configured refiner requires refiner state in the resume checkpoint.
    Use stage_2.ckpt for a base-model warm start.
    """
    checkpoint = torch.load(path, map_location="cpu")
    if "model" not in checkpoint or "ema" not in checkpoint:
        raise KeyError(
            f"{path} is not a Stage-2 training checkpoint (keys: {sorted(checkpoint)[:8]}). "
            "Anything matching 'ep-<digits>' in the checkpoint dir is treated as a resume "
            "candidate; move stray weight dumps out of it."
        )
    if refiner is not None and checkpoint.get("refiner") is None:
        raise KeyError(
            f"{path} has no 'refiner' state for joint-training resume. "
            "Use a joint-training checkpoint, or use stage_2.ckpt for a base-model warm start."
        )
    _unwrap(model).load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if refiner is not None and checkpoint.get("refiner") is not None:
        _unwrap(refiner).load_state_dict(checkpoint["refiner"])
        if refiner_ema is not None and checkpoint.get("refiner_ema") is not None:
            refiner_ema.load_state_dict(checkpoint["refiner_ema"])
        if refiner_optimizer is not None and checkpoint.get("refiner_optimizer") is not None:
            refiner_optimizer.load_state_dict(checkpoint["refiner_optimizer"])
        if refiner_scheduler is not None and checkpoint.get("refiner_scheduler") is not None:
            refiner_scheduler.load_state_dict(checkpoint["refiner_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


__all__ = [
    "save_stage1_checkpoint",
    "load_stage1_checkpoint",
    "save_stage2_checkpoint",
    "load_stage2_checkpoint",
    "save_refiner_weights",
]
