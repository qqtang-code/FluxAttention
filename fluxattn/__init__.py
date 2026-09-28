# Core sparse attention kernels
try:
    from .kernels import (
        Xattention_prefill_dim3,
        Xattention_prefill_dim4,
    )
except ImportError as exc:  # block_sparse_attn / triton are CUDA-only
    _KERNEL_IMPORT_ERROR = exc

    def __getattr__(name):
        if name in ("Xattention", "Xattention_prefill_dim3", "Xattention_prefill_dim4"):
            raise ImportError(
                f"fluxattn.{name} needs the sparse attention extensions "
                "(block_sparse_attn, triton), which are not importable here. "
                "Install them to use the XAttention kernels; fluxattn.batch only "
                "needs PyTorch."
            ) from _KERNEL_IMPORT_ERROR
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    __all__ = []
else:
    # For backward compatibility and ease of use
    Xattention = Xattention_prefill_dim4

    __all__ = [
        # Kernels
        "Xattention_prefill_dim3",
        "Xattention_prefill_dim4",
        # Aliases for backward compatibility
        "Xattention",
    ]