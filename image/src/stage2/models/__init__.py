"""Encoder registry for stage 2 models."""
from typing import Any, Protocol, runtime_checkable

from .self_flow_dit import SelfFlowDiT


@runtime_checkable
class Stage2ModelProtocol(Protocol):
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...


__all__ = ["SelfFlowDiT", "Stage2ModelProtocol"]
