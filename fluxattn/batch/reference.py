from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .config import StreamingConfig


def reference_batch_flux_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    route: Tensor,
    cfg: StreamingConfig,
    q_offset: Optional[int] = None,
) -> Tensor:
    """Materialises the routed attention pattern with plain softmax attention.

    Kept as a slow, obviously-correct oracle for the FlexAttention kernel.
    Inputs follow the FlexAttention layout, ``[B, H, S, D]``, and ``q_offset``
    defaults to tail alignment exactly like ``batch_flux_attention``.
    """
    B, H, S, D = q.shape
    Bk, Hk, Sk, Dk = k.shape
    if Hk != H:
        expand = H // Hk
        k = k.repeat_interleave(expand, dim=1)
        v = v.repeat_interleave(expand, dim=1)

    if q_offset is None:
        q_offset = Sk - S
    if S > Sk or q_offset < 0 or q_offset + S > Sk:
        raise ValueError(
            f"query block [{q_offset}, {q_offset + S}) does not fit in {Sk} key positions"
        )

    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    q_idx = torch.arange(q_offset, q_offset + S, device=q.device).view(1, 1, S, 1)
    k_idx = torch.arange(Sk, device=q.device).view(1, 1, 1, Sk)

    in_window = k_idx > (q_idx - cfg.window)
    in_sink = k_idx < cfg.sink
    stream_ok = in_window | in_sink
    is_stream = (route != 0).view(B, 1, 1, 1)
    allowed = stream_ok | (~is_stream)
    if cfg.causal:
        allowed = allowed & (q_idx >= k_idx)

    scores = scores.masked_fill(~allowed, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
