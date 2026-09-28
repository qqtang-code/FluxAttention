"""Sparse attention kernels.

- XAttention: adaptive block-sparse prefill based on thresholding.
- Streaming attention (sink + sliding window) comes from ``block_sparse_attn``.

The public entry points are ``Xattention_prefill_dim3`` (unpadded, packed
sequences) and ``Xattention_prefill_dim4`` (padded, per-sample sequences);
``Xattention`` is kept as an alias of the dim4 variant.
"""

from .xattention import Xattention_prefill_dim3, Xattention_prefill_dim4

__all__ = [
    "Xattention_prefill_dim3",
    "Xattention_prefill_dim4",
]