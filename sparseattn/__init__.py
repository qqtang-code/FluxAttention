# Core sparse attention implementations
from .src import (
    Xattention_prefill_dim3,
    Xattention_prefill_dim4,
)


# For backward compatibility and ease of use
Xattention = Xattention_prefill_dim4

__all__ = [
    # Core modules
    "Xattention_prefill",
    # Training modules
    # Aliases for backward compatibility
    "Xattention",
]
