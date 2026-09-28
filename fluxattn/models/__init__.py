"""Model definitions for inference.

These are the transformers models shipped with Flux Attention: they add the
layer router and the hybrid attention path on top of the backbone, and support
``generate`` with a KV cache. Load them with ``trust_remote_code=True`` (see
the README) or register them manually:

    from fluxattn.models.modeling_qwen3 import PawQwen3Config, PawQwen3ForCausalLM

The training-time variants live in :mod:`fluxattn.training.modeling`.
"""

__all__ = ["modeling_llama", "modeling_qwen3"]