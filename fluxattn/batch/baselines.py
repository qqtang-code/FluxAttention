from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .config import StreamingConfig

try:
    from flash_attn import flash_attn_func as _fa_func
except Exception:
    _fa_func = None

try:
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_func as _fa_varlen_func,
    )
except Exception:
    _fa_varlen_func = None


def _fa_with_lse(
    q_bshd: Tensor,
    k_bshd: Tensor,
    v_bshd: Tensor,
    causal: bool,
    window_size: tuple[int, int] = (-1, -1),
) -> tuple[Tensor, Tensor]:
    out, lse, _ = _fa_func(
        q_bshd,
        k_bshd,
        v_bshd,
        causal=causal,
        window_size=window_size,
        return_attn_probs=True,
    )
    return out, lse


def flash_attn_split_window_only(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    route: Tensor,
    cfg: StreamingConfig,
) -> Tensor:
    """Baseline that groups the batch and launches FA2 twice.

    Dropped the attention sink whenever ``cfg.sink > 0``, so it is only a speed
    reference, not a numerically equivalent implementation.
    """
    if _fa_func is None:
        raise RuntimeError("flash-attn is not installed; install `flash-attn>=2.5`.")

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()

    route = route.to(torch.bool)
    dense_idx = torch.nonzero(~route, as_tuple=False).squeeze(-1)
    stream_idx = torch.nonzero(route, as_tuple=False).squeeze(-1)

    out = torch.empty_like(q_bshd)

    if dense_idx.numel() > 0:
        qd = q_bshd.index_select(0, dense_idx)
        kd = k_bshd.index_select(0, dense_idx)
        vd = v_bshd.index_select(0, dense_idx)
        od = _fa_func(qd, kd, vd, causal=cfg.causal)
        out.index_copy_(0, dense_idx, od)

    if stream_idx.numel() > 0:
        qs = q_bshd.index_select(0, stream_idx)
        ks = k_bshd.index_select(0, stream_idx)
        vs = v_bshd.index_select(0, stream_idx)
        os_ = _fa_func(
            qs,
            ks,
            vs,
            causal=cfg.causal,
            window_size=(
                (cfg.window - 1, 0) if cfg.causal else (cfg.window - 1, cfg.window - 1)
            ),
        )
        out.index_copy_(0, stream_idx, os_)

    return out.transpose(1, 2).contiguous()


def flash_attn_full_dense(
    q: Tensor, k: Tensor, v: Tensor, cfg: StreamingConfig
) -> Tensor:
    if _fa_func is None:
        raise RuntimeError("flash-attn is not installed.")
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    o = _fa_func(q, k, v, causal=cfg.causal)
    return o.transpose(1, 2).contiguous()


def has_flash_attn() -> bool:
    return _fa_func is not None
