# Core sparse attention implementations
try:
    from .src import (
        Xattention_prefill_dim3,
        Xattention_prefill_dim4,
    )
except ImportError as exc:  # block_sparse_attn / triton are CUDA-only
    _SPARSE_IMPORT_ERROR = exc

    def __getattr__(name):
        if name in ("Xattention", "Xattention_prefill_dim3", "Xattention_prefill_dim4"):
            raise ImportError(
                f"fluxattn.{name} needs the sparse attention extensions "
                "(block_sparse_attn, triton), which are not importable here. "
                "Install them to use the XAttention kernels; fluxattn.batch only "
                "needs PyTorch."
            ) from _SPARSE_IMPORT_ERROR
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    __all__ = []
else:
    # For backward compatibility and ease of use
    Xattention = Xattention_prefill_dim4

    __all__ = [
        # Core modules
        "Xattention_prefill_dim3",
        "Xattention_prefill_dim4",
        # Aliases for backward compatibility
        "Xattention",
    ]