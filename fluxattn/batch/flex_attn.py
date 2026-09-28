from __future__ import annotations

from typing import Callable, Optional

import torch
import torch._dynamo
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from .config import StreamingConfig

# Each distinct (query length, key/value length, block size) triple lands in its
# own Dynamo graph, so a batch of long-context requests exhausts the default
# limit quickly. max() keeps any user-raised limit intact.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)

_compiled_flex = torch.compile(flex_attention, dynamic=False, fullgraph=True)
_compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=False, fullgraph=True
)


def _make_mask_mod(route: Tensor, cfg: StreamingConfig, q_offset: int) -> Callable:
    window = cfg.window
    sink = cfg.sink
    causal = cfg.causal

    def mask_mod(b, h, q_idx, kv_idx):
        pos = q_idx + q_offset
        is_stream = route[b] != 0
        in_window = kv_idx > pos - window
        in_sink = kv_idx < sink
        stream_ok = in_window | in_sink
        allowed = stream_ok | (~is_stream)
        if causal:
            allowed = allowed & (pos >= kv_idx)
        return allowed

    return mask_mod


def batch_flux_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    route: Tensor,
    cfg: StreamingConfig,
    block_size: int = 128,
    q_offset: Optional[int] = None,
) -> Tensor:
    """Routed dense/streaming attention in a single FlexAttention launch.

    Layout is FlexAttention native, ``[B, H, S, D]``, so ``H`` may differ
    between ``q`` and ``k``/``v`` (GQA broadcasts automatically). ``route`` is a
    ``[B]`` integer tensor where ``0`` selects dense attention and any non-zero
    value selects streaming attention (``cfg``).

    Query and key/value lengths may differ. ``q_offset`` is the absolute
    position of the first query token inside the key sequence; it defaults to
    ``Sk - S``, i.e. the query block sits at the tail of the keys, which covers
    prefill (``S == Sk``, offset 0) and decode against a cache (``S == 1``).
    Chunked prefill against a longer cache must pass the offset explicitly.

    Each distinct ``(S, Sk, q_offset)`` combination builds its own block mask,
    so a decode loop that grows the cache by one token at a time rebuilds the
    mask on every step.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be 4-D [B, H, S, D]")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q/k/v must be contiguous in the head-dim dimension")

    B, H, S, D = q.shape
    Bk, Hk, Sk, Dk = k.shape
    if (B, D) != (Bk, Dk):
        raise ValueError("batch and head-dim of q and k must match")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if H % Hk != 0:
        raise ValueError(f"num query heads ({H}) must be a multiple of kv heads ({Hk})")
    if route.shape != (B,):
        raise ValueError(f"route must have shape ({B},), got {tuple(route.shape)}")
    if route.device != q.device:
        raise ValueError("route must live on the same device as q")

    if q_offset is None:
        q_offset = Sk - S
    if S > Sk or q_offset < 0 or q_offset + S > Sk:
        raise ValueError(
            f"query block [{q_offset}, {q_offset + S}) does not fit in {Sk} key positions"
        )

    route_i32 = route.to(torch.int32).contiguous()
    mask_mod = _make_mask_mod(route_i32, cfg, int(q_offset))

    block_mask = _compiled_create_block_mask(
        mask_mod,
        B=B,
        H=None,
        Q_LEN=S,
        KV_LEN=Sk,
        device=q.device,
        BLOCK_SIZE=block_size,
    )

    enable_gqa = Hk != H
    return _compiled_flex(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)