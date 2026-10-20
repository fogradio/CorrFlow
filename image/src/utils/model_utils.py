"""Model instantiation utilities."""

import importlib

import torch

from configs import ModelConfig


def get_obj_from_str(string: str, reload: bool = False):
    """Import and return a class/function from a dotted string path."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: ModelConfig) -> object:
    """Instantiate a model from ModelConfig.

    Args:
        config: ModelConfig with target, params, and optional ckpt path

    Returns:
        Instantiated model, optionally with loaded checkpoint
    """
    if not config.target:
        raise KeyError("Expected 'target' to instantiate.")

    model = get_obj_from_str(config.target)(**config.params)

    if getattr(config, "ckpt", None) is not None:
        ckpt_key = getattr(config, "ckpt_key", None)
        # mmap only pages in the tensors load_state_dict actually copies -- a
        # stage-2 training checkpoint also carries the fast weights, the EMA and
        # both optimizer states, so reading it whole costs ~11.5 GB per rank.
        state_dict = torch.load(config.ckpt, map_location="cpu", mmap=True)
        if ckpt_key is not None:
            if ckpt_key not in state_dict:
                raise KeyError(
                    f"ckpt_key='{ckpt_key}' not in {config.ckpt} "
                    f"(keys: {sorted(k for k in state_dict)})"
                )
            state_dict = state_dict[ckpt_key]
        elif "ema" in state_dict:
            state_dict = state_dict["ema"]
        elif "model" in state_dict:
            raise NotImplementedError("Loading from 'model' key not implemented yet.")
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded {config.target} from {config.ckpt}"
              + (f" [key={ckpt_key}]" if ckpt_key is not None else ""))

    return model
