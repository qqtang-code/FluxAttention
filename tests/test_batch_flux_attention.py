from __future__ import annotations

import math

import pytest
import torch

from fluxattn.batch import (
    BatchedRoutedAttention,
    StreamingConfig,
    batch_flux_attention,
    reference_batch_flux_attention,
    route_from_sparse_mask,
)


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("FlexAttention requires CUDA")


# --------------------------------------------------------------------------- #
# Router plumbing (runs on CPU)
# --------------------------------------------------------------------------- #


def test_route_from_sparse_mask_follows_router_convention() -> None:
    # The router emits z=1 for the exact path and z=0 for streaming, replicated
    # over heads.
    z = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(4, 1, 1).expand(4, 3, 1)
    route = route_from_sparse_mask(z)
    assert route.dtype == torch.int32
    assert route.tolist() == [0, 1, 0, 1]


def test_route_from_sparse_mask_flattens_per_sample_heads() -> None:
    z = torch.ones(3, 8, 1)
    z[1] = 0.0
    z[2, :4] = 0.0  # a partially streamed sample falls back to majority vote
    assert route_from_sparse_mask(z).tolist() == [0, 1, 1]


def test_route_from_sparse_mask_rejects_scalars() -> None:
    with pytest.raises(ValueError):
        route_from_sparse_mask(torch.tensor(1.0))


def test_stable_route_buffer_is_reused_across_calls() -> None:
    # Re-using the same tensor keeps the compiled block-mask builder's guards
    # valid; a fresh tensor per forward would recompile every step.
    runner = BatchedRoutedAttention(window=8, sink=1)
    first = runner._stable_route(torch.tensor([0, 1, 1]))
    second = runner._stable_route(torch.tensor([1, 0, 0]))
    assert first is second
    assert second.tolist() == [1, 0, 0]

    other_batch = runner._stable_route(torch.tensor([1, 1]))
    assert other_batch.numel() == 2


def test_streaming_config_validates_itself() -> None:
    with pytest.raises(ValueError):
        StreamingConfig(window=0)
    with pytest.raises(ValueError):
        StreamingConfig(window=4, sink=-1)


# --------------------------------------------------------------------------- #
# Routed mask semantics, checked against closed-form expectations (CPU)
# --------------------------------------------------------------------------- #


def _tensors(B: int, H: int, S: int, D: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, H, S, D, generator=g) for _ in range(3))


def test_window_of_one_makes_streaming_a_pure_copy() -> None:
    # window=1 keeps only kv_idx == q_idx, so the output is v regardless of k/q.
    B, H, S, D = 3, 2, 32, 16
    q, k, v = _tensors(B, H, S, D)
    cfg = StreamingConfig(window=1, sink=0, causal=True)
    route = torch.ones(B, dtype=torch.int32)
    out = reference_batch_flux_attention(q, k, v, route, cfg)
    torch.testing.assert_close(out, v, atol=1e-6, rtol=1e-6)


def test_sink_only_attends_to_the_first_tokens() -> None:
    B, H, S, D = 2, 2, 16, 8
    q, k, v = _tensors(B, H, S, D, seed=1)
    cfg = StreamingConfig(window=1, sink=2, causal=True)
    route = torch.ones(B, dtype=torch.int32)
    out = reference_batch_flux_attention(q, k, v, route, cfg)

    # The very first query sees only the sink, whose rows are exact v rows.
    torch.testing.assert_close(out[:, :, 0], v[:, :, 0], atol=1e-6, rtol=1e-6)

    # For query 1 the allowed keys are {0, 1}, i.e. the sink plus itself.
    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("bhd,bhkd->bhk", q[:, :, 1], k[:, :, :2]) * scale
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("bhk,bhkd->bhd", probs, v[:, :, :2])
    torch.testing.assert_close(out[:, :, 1], expected, atol=1e-6, rtol=1e-6)


def test_wide_window_reproduces_causal_attention() -> None:
    B, H, S, D = 2, 2, 24, 16
    q, k, v = _tensors(B, H, S, D, seed=2)
    cfg = StreamingConfig(window=S, sink=0, causal=True)
    route = torch.ones(B, dtype=torch.int32)
    out = reference_batch_flux_attention(q, k, v, route, cfg)
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, dense, atol=1e-5, rtol=1e-5)


def test_route_selects_the_pattern_per_sample() -> None:
    B, H, S, D = 2, 2, 32, 16
    q, k, v = _tensors(B, H, S, D, seed=3)
    cfg = StreamingConfig(window=4, sink=1, causal=True)
    route = torch.tensor([0, 1], dtype=torch.int32)
    out = reference_batch_flux_attention(q, k, v, route, cfg)

    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out[0], dense[0], atol=1e-6, rtol=1e-6)

    streamed = reference_batch_flux_attention(
        q[1:], k[1:], v[1:], torch.ones(1, dtype=torch.int32), cfg
    )
    torch.testing.assert_close(out[1], streamed[0], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out[1], dense[1], atol=1e-3)


def test_reference_supports_gqa() -> None:
    B, Hq, Hkv, S, D = 2, 4, 2, 32, 16
    g = torch.Generator().manual_seed(4)
    q = torch.randn(B, Hq, S, D, generator=g)
    k = torch.randn(B, Hkv, S, D, generator=g)
    v = torch.randn(B, Hkv, S, D, generator=g)
    cfg = StreamingConfig(window=S, sink=0, causal=True)
    route = torch.zeros(B, dtype=torch.int32)

    out = reference_batch_flux_attention(q, k, v, route, cfg)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True
    )
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_dense_samples_use_query_and_key_lengths_independently() -> None:
    # Decode-shaped input: one query token against a longer cache. The query is
    # tail aligned, so causality leaves the whole cache visible.
    B, H, S, Sk, D = 2, 2, 1, 16, 16
    g = torch.Generator().manual_seed(5)
    q = torch.randn(B, H, S, D, generator=g)
    k = torch.randn(B, H, Sk, D, generator=g)
    v = torch.randn(B, H, Sk, D, generator=g)
    cfg = StreamingConfig(window=4, sink=0, causal=True)
    route = torch.zeros(B, dtype=torch.int32)

    out = reference_batch_flux_attention(q, k, v, route, cfg)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)
    probs = torch.softmax(scores, dim=-1)
    torch.testing.assert_close(out, torch.matmul(probs, v), atol=1e-6, rtol=1e-6)


def test_decode_shape_anchors_the_window_at_the_query_position() -> None:
    # A tail-aligned query at position Sk-1 must see the last `window` keys, not
    # the first ones.
    B, H, Sk, D = 1, 2, 32, 8
    g = torch.Generator().manual_seed(8)
    q = torch.randn(B, H, 1, D, generator=g)
    k = torch.randn(B, H, Sk, D, generator=g)
    v = torch.randn(B, H, Sk, D, generator=g)
    cfg = StreamingConfig(window=4, sink=0, causal=True)
    route = torch.ones(B, dtype=torch.int32)

    out = reference_batch_flux_attention(q, k, v, route, cfg)
    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("bhd,bhkd->bhk", q[:, :, 0], k[:, :, -4:]) * scale
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("bhk,bhkd->bhd", probs, v[:, :, -4:])
    torch.testing.assert_close(out[:, :, 0], expected, atol=1e-6, rtol=1e-6)


def test_query_offset_can_be_given_explicitly() -> None:
    B, H, S, Sk, D = 1, 1, 2, 8, 8
    g = torch.Generator().manual_seed(9)
    q = torch.randn(B, H, S, D, generator=g)
    k = torch.randn(B, H, Sk, D, generator=g)
    v = torch.randn(B, H, Sk, D, generator=g)
    cfg = StreamingConfig(window=Sk, sink=0, causal=True)
    route = torch.zeros(B, dtype=torch.int32)

    # A two-token chunk that ends at position 5 sees keys 0..5.
    out = reference_batch_flux_attention(q, k, v, route, cfg, q_offset=4)
    scores = torch.matmul(q, k[:, :, :6].transpose(-1, -2)) / math.sqrt(D)
    qi = torch.arange(4, 6).view(1, 1, 2, 1)
    ki = torch.arange(6).view(1, 1, 1, 6)
    probs = torch.softmax(scores.masked_fill(qi < ki, float("-inf")), dim=-1)
    torch.testing.assert_close(
        out, torch.matmul(probs, v[:, :, :6]), atol=1e-6, rtol=1e-6
    )


def test_query_offset_is_validated() -> None:
    q, k, v = _tensors(1, 1, 2, 8)
    cfg = StreamingConfig(window=4)
    route = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError):
        reference_batch_flux_attention(q, k, v, route, cfg, q_offset=99)
    with pytest.raises(ValueError):
        batch_flux_attention(q, k, v, route, cfg, q_offset=99)


# --------------------------------------------------------------------------- #
# Kernel argument validation (CPU: raises before any CUDA launch)
# --------------------------------------------------------------------------- #


def test_kernel_validates_route_shape_and_device() -> None:
    q, k, v = _tensors(2, 4, 8, 16)
    cfg = StreamingConfig(window=4)
    with pytest.raises(ValueError):
        batch_flux_attention(q, k, v, torch.zeros(3, dtype=torch.int32), cfg)
    with pytest.raises(ValueError):
        batch_flux_attention(
            q, k, v, torch.zeros(2, dtype=torch.int32, device="meta"), cfg
        )


def test_kernel_validates_head_counts() -> None:
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 3, 8, 16)
    v = torch.randn(2, 3, 8, 16)
    cfg = StreamingConfig(window=4)
    with pytest.raises(ValueError):
        batch_flux_attention(q, k, v, torch.zeros(2, dtype=torch.int32), cfg)


# --------------------------------------------------------------------------- #
# Model layout adapter, exercised with the reference standing in for the kernel
# --------------------------------------------------------------------------- #


def _install_reference_kernel(monkeypatch) -> dict:
    """Replaces the FlexAttention call so the adapter can run on CPU."""
    import fluxattn.batch.modeling as modeling

    seen = {}

    def fake_kernel(q, k, v, route, cfg, block_size=128, q_offset=None):
        seen["q_shape"] = tuple(q.shape)
        seen["k_shape"] = tuple(k.shape)
        seen["route"] = route.tolist()
        seen["q_offset"] = q_offset
        return reference_batch_flux_attention(q, k, v, route, cfg, q_offset=q_offset)

    monkeypatch.setattr(modeling, "batch_flux_attention", fake_kernel)
    return seen


def test_model_layout_adapter_round_trips_bshd(monkeypatch) -> None:
    seen = _install_reference_kernel(monkeypatch)
    runner = BatchedRoutedAttention(window=4, sink=1)

    B, S, Hq, Hkv, D = 2, 16, 4, 2, 16
    g = torch.Generator().manual_seed(10)
    q = torch.randn(B, S, Hq, D, generator=g)
    k = torch.randn(B, S, Hkv, D, generator=g)
    v = torch.randn(B, S, Hkv, D, generator=g)
    route = torch.tensor([0, 1], dtype=torch.int32)

    out = runner(q, k, v, route)

    assert out.shape == q.shape
    assert seen["q_shape"] == (B, Hq, S, D)
    assert seen["k_shape"] == (B, Hkv, S, D)
    assert seen["route"] == [0, 1]
    assert seen["q_offset"] is None

    # The dense sample must reproduce causal SDPA with GQA broadcast.
    dense = torch.nn.functional.scaled_dot_product_attention(
        q[0:1].transpose(1, 2),
        k[0:1].transpose(1, 2),
        v[0:1].transpose(1, 2),
        is_causal=True,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(out[0], dense[0], atol=1e-5, rtol=1e-5)


def test_model_layout_adapter_handles_decode_shapes(monkeypatch) -> None:
    seen = _install_reference_kernel(monkeypatch)
    runner = BatchedRoutedAttention(window=4, sink=0)

    B, Sk, Hq, Hkv, D = 2, 32, 4, 4, 16
    g = torch.Generator().manual_seed(11)
    q = torch.randn(B, 1, Hq, D, generator=g)
    k = torch.randn(B, Sk, Hkv, D, generator=g)
    v = torch.randn(B, Sk, Hkv, D, generator=g)
    route = torch.zeros(B, dtype=torch.int32)

    out = runner(q, k, v, route)

    assert out.shape == q.shape
    assert seen["q_shape"] == (B, Hq, 1, D)
    assert seen["k_shape"] == (B, Hkv, Sk, D)

    # Tail alignment: the single query sees the whole cache.
    scores = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-1, -2))
    probs = torch.softmax(scores / math.sqrt(D), dim=-1)
    expected = torch.matmul(probs, v.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# FlexAttention kernel against the reference (CUDA only)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("sink", [0, 4])
def test_mixed_route_matches_reference(
    dtype: torch.dtype, causal: bool, sink: int
) -> None:
    _skip_if_no_cuda()
    torch.manual_seed(0)
    device = "cuda"

    B, H, S, D = 4, 4, 256, 64
    window = 64
    cfg = StreamingConfig(window=window, sink=sink, causal=causal)

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    route = torch.tensor([0, 1, 1, 0], device=device, dtype=torch.int32)

    out = batch_flux_attention(q, k, v, route, cfg)
    ref = reference_batch_flux_attention(q.float(), k.float(), v.float(), route, cfg)

    atol = 1e-4 if dtype == torch.float32 else 5e-3
    rtol = 1e-4 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)


def test_all_dense_matches_sdpa() -> None:
    _skip_if_no_cuda()
    torch.manual_seed(1)
    device = "cuda"

    B, H, S, D = 2, 8, 512, 64
    cfg = StreamingConfig(window=64, sink=0, causal=True)

    q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    route = torch.zeros(B, device=device, dtype=torch.int32)

    out = batch_flux_attention(q, k, v, route, cfg)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out.float(), sdpa.float(), atol=5e-3, rtol=5e-3)


def test_all_streaming_matches_reference() -> None:
    _skip_if_no_cuda()
    torch.manual_seed(2)
    device = "cuda"

    B, H, S, D = 3, 4, 384, 64
    cfg = StreamingConfig(window=48, sink=2, causal=True)

    q = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    route = torch.ones(B, device=device, dtype=torch.int32)

    out = batch_flux_attention(q, k, v, route, cfg)
    ref = reference_batch_flux_attention(q, k, v, route, cfg)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_gqa() -> None:
    _skip_if_no_cuda()
    torch.manual_seed(3)
    device = "cuda"

    B, Hq, Hkv, S, D = 2, 8, 2, 256, 64
    cfg = StreamingConfig(window=32, sink=0, causal=True)

    q = torch.randn(B, Hq, S, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, S, D, device=device, dtype=torch.bfloat16)
    route = torch.tensor([1, 0], device=device, dtype=torch.int32)

    out = batch_flux_attention(q, k, v, route, cfg)
    ref = reference_batch_flux_attention(q.float(), k.float(), v.float(), route, cfg)
    torch.testing.assert_close(out.float(), ref, atol=5e-3, rtol=5e-3)


def test_decode_shape_matches_reference() -> None:
    _skip_if_no_cuda()
    torch.manual_seed(6)
    device = "cuda"

    B, H, Sk, D = 4, 4, 512, 64
    cfg = StreamingConfig(window=64, sink=8, causal=True)

    q = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, Sk, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, Sk, D, device=device, dtype=torch.float32)
    route = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.int32)

    out = batch_flux_attention(q, k, v, route, cfg)
    ref = reference_batch_flux_attention(q, k, v, route, cfg)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_runner_matches_functional_api() -> None:
    _skip_if_no_cuda()
    torch.manual_seed(7)
    device = "cuda"

    B, H, S, D = 3, 4, 256, 64
    cfg = StreamingConfig(window=32, sink=4, causal=True)
    runner = BatchedRoutedAttention(window=cfg.window, sink=cfg.sink)

    q = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    route = torch.tensor([1, 0, 1], device=device, dtype=torch.int32)

    # The runner consumes [B, S, H, D]; the functional API consumes [B, H, S, D].
    q_bshd, k_bshd, v_bshd = (t.transpose(1, 2) for t in (q, k, v))
    out = runner(q_bshd, k_bshd, v_bshd, route).transpose(1, 2)
    ref = batch_flux_attention(q, k, v, route, cfg)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    # A second call with a fresh route tensor must produce a different result.
    out2 = runner(
        q_bshd, k_bshd, v_bshd, torch.zeros(B, device=device, dtype=torch.int32)
    )
    assert not torch.allclose(out2.transpose(1, 2), out, atol=1e-4)
