"""Benchmarks per-sample routed attention against dense baselines.

Prefill (default) and decode (``--qlen 1``) shapes are both supported. The
benchmark is only meaningful when the batch mixes dense and streaming samples;
``--ratio`` controls how many samples are routed to streaming.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from fluxattn.batch import StreamingConfig, batch_flux_attention
from fluxattn.batch.baselines import (
    flash_attn_full_dense,
    flash_attn_split_window_only,
    has_flash_attn,
)


@dataclass
class BenchResult:
    name: str
    ms: float


def _bench(
    fn: Callable[[], torch.Tensor], warmup: int = 3, iters: int = 20
) -> BenchResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return BenchResult(name="", ms=ms)


def run(
    B: int,
    H: int,
    S: int,
    D: int,
    window: int,
    sink: int,
    ratio: float,
    qlen: Optional[int] = None,
) -> None:
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)
    cfg = StreamingConfig(window=window, sink=sink, causal=True)

    kv_len = S
    q_len = S if qlen is None else qlen

    q = torch.randn(B, H, q_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, kv_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, kv_len, D, device=device, dtype=dtype)

    n_stream = int(round(B * ratio))
    perm = torch.randperm(B, device=device)
    route = torch.zeros(B, device=device, dtype=torch.int32)
    route[perm[:n_stream]] = 1

    results: List[BenchResult] = []

    r = _bench(lambda: batch_flux_attention(q, k, v, route, cfg))
    r.name = "FlexAttention (routed)"
    results.append(r)

    def _sdpa_dense() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    r = _bench(_sdpa_dense)
    r.name = "SDPA (all dense)"
    results.append(r)

    if has_flash_attn():
        r = _bench(lambda: flash_attn_full_dense(q, k, v, cfg))
        r.name = "FA2 (all dense)"
        results.append(r)

        r = _bench(lambda: flash_attn_split_window_only(q, k, v, route, cfg))
        r.name = "FA2 split (window-only*)"
        results.append(r)
    else:
        print("[info] flash-attn not installed, skipping FA2 baselines")

    baseline = next((x.ms for x in results if x.name == "FA2 (all dense)"), None)
    if baseline is None:
        baseline = next(x.ms for x in results if x.name == "SDPA (all dense)")

    mode = "decode" if q_len == 1 else "prefill"
    print(
        f"\n[{mode}] B={B} H={H} Q_LEN={q_len} KV_LEN={kv_len} D={D}  "
        f"window={window} sink={sink}  stream_ratio={n_stream}/{B}"
    )
    print(f"{'kernel':<28s} {'ms':>10s} {'vs dense':>10s}")
    print("-" * 50)
    for r in results:
        print(f"{r.name:<28s} {r.ms:>10.3f} {baseline / r.ms:>9.2f}x")
    if sink > 0 and has_flash_attn():
        print("* FA2 split drops the attention-sink portion (not numerically equivalent).")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--S", type=int, default=4096, help="key/value sequence length")
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument(
        "--qlen",
        type=int,
        default=None,
        help="query length; defaults to --S (prefill). Use 1 for decode.",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="fraction of batches routed to streaming attention",
    )
    args = p.parse_args()
    run(args.B, args.H, args.S, args.D, args.window, args.sink, args.ratio, args.qlen)


if __name__ == "__main__":
    main()