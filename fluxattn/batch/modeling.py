from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from .config import StreamingConfig
from .flex_attn import batch_flux_attention

DENSE: int = 0
STREAMING: int = 1

__all__ = [
    "DENSE",
    "STREAMING",
    "BatchedRoutedAttention",
    "batched_routed_attention",
    "route_from_sparse_mask",
]


def route_from_sparse_mask(
    sparse_mask: Tensor, dense_when_one: bool = True
) -> Tensor:
    """Turns the layer router output into the per-sample route the kernel wants.

    ``sparse_mask`` is the router's ``z`` with shape ``[B, H, 1]`` (or any shape
    whose leading dim is the batch), replicated across heads. ``dense_when_one``
    describes the router convention: the Flux Attention ``AttentionRouter``
    initialises its bias towards the exact path, so ``z == 1`` means dense and
    ``z == 0`` means streaming.

    Returns an ``int32`` tensor of shape ``[B]`` where ``0`` is dense and ``1``
    is streaming.
    """
    if sparse_mask.dim() == 0:
        raise ValueError("sparse_mask must have at least one dimension")
    z = sparse_mask.reshape(sparse_mask.shape[0], -1)
    if z.size(1) > 1:
        # Heads normally carry the same decision; average in case they do not.
        z = z.mean(dim=1, keepdim=True)
    z = z[:, 0]
    wants_dense = z > 0.5 if dense_when_one else z <= 0.5
    return wants_dense.logical_not().to(torch.int32).contiguous()


def batched_routed_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    route: Tensor,
    cfg: StreamingConfig,
    block_size: int = 128,
    q_offset: Optional[int] = None,
) -> Tensor:
    """Routed attention in the model layout: ``q`` is ``[B, S, H, D]``.

    ``k``/``v`` are ``[B, Sk, H_kv, D]`` and may hold fewer heads than ``q``
    (GQA). ``S`` and ``Sk`` may differ: prefill has ``S == Sk`` and decode has
    ``S == 1`` with ``Sk`` the cache length. ``q_offset`` defaults to tail
    alignment, which is what both of those cases want.
    """
    if q.dim() != 4:
        raise ValueError("q must be 4-D [B, S, H, D]")
    out = batch_flux_attention(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        route,
        cfg,
        block_size=block_size,
        q_offset=q_offset,
    )
    return out.transpose(1, 2)


class BatchedRoutedAttention:
    """Per-sample dense/streaming attention for a batch of requests.

    The layer router decides independently, for every sequence in the batch,
    whether that sequence can be approximated with streaming attention. This
    wrapper feeds those decisions to FlexAttention so the whole batch is served
    by one kernel launch instead of running both branches over every sample.

    ``torch.compile`` guards the block-mask builder on the tensors captured by
    the mask closure. The route tensor is therefore copied into a buffer that
    persists per (device, batch size): passing a freshly allocated ``route`` on
    every forward would otherwise invalidate the guard and recompile the mask
    builder on each step.

    ``head_dim`` must be supported by FlexAttention (a multiple of 16 is the
    safe assumption), and the device must be CUDA.
    """

    def __init__(
        self,
        window: int,
        sink: int = 0,
        causal: bool = True,
        block_size: int = 128,
    ) -> None:
        self.cfg = StreamingConfig(window=int(window), sink=int(sink), causal=bool(causal))
        self.block_size = int(block_size)
        self._route_buffers: Dict[Tuple[torch.device, int], Tensor] = {}

    @classmethod
    def from_model_config(
        cls, config: Any, causal: bool = True, block_size: int = 128
    ) -> "BatchedRoutedAttention":
        cfg = StreamingConfig.from_model_config(config, causal=causal)
        return cls(cfg.window, cfg.sink, cfg.causal, block_size=block_size)

    @property
    def window(self) -> int:
        return self.cfg.window

    @property
    def sink(self) -> int:
        return self.cfg.sink

    def _stable_route(self, route: Tensor) -> Tensor:
        if route.dim() != 1:
            raise ValueError(f"route must be 1-D [B], got {tuple(route.shape)}")
        key = (route.device, route.numel())
        buffer = self._route_buffers.get(key)
        if buffer is None:
            buffer = torch.empty(route.numel(), dtype=torch.int32, device=route.device)
            self._route_buffers[key] = buffer
        buffer.copy_(route.to(torch.int32))
        return buffer

    def __call__(
        self, q: Tensor, k: Tensor, v: Tensor, route: Tensor, q_offset: Optional[int] = None
    ) -> Tensor:
        return batched_routed_attention(
            q,
            k,
            v,
            self._stable_route(route),
            self.cfg,
            block_size=self.block_size,
            q_offset=q_offset,
        )