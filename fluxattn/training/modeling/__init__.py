"""Training-time model definitions.

Same architectures as :mod:`fluxattn.models`, with the forward contract the
trainer needs: routing auxiliary outputs (decisions, entropy, pooled features)
and the sparse/dense interpolation, without the KV-cache path.
"""

from .modeling_llama import PawLlamaConfig, PawLlamaForCausalLM
from .modeling_qwen3 import PawQwen3Config, PawQwen3ForCausalLM

__all__ = [
    "PawLlamaConfig",
    "PawLlamaForCausalLM",
    "PawQwen3Config",
    "PawQwen3ForCausalLM",
]