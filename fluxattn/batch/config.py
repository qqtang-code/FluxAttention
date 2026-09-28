from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StreamingConfig:
    """Streaming attention pattern: attention sink + sliding window.

    A token attends to the first ``sink`` tokens of the sequence (attention
    sinks) union the most recent ``window`` tokens, optionally intersected with
    the causal mask.

    ``window`` and ``sink`` are counted in tokens, not in blocks: the mask is
    built on block granularity but FlexAttention ``mask_mod`` is evaluated per
    element, so no rounding is introduced.
    """

    window: int
    sink: int = 0
    causal: bool = True

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.sink < 0:
            raise ValueError("sink must be non-negative")

    @classmethod
    def from_model_config(cls, config: Any, causal: bool = True) -> "StreamingConfig":
        """Builds the streaming pattern from a Flux Attention model config."""
        return cls(
            window=int(getattr(config, "local_window_size")),
            sink=int(getattr(config, "sink_size", 0)),
            causal=causal,
        )
