"""Training pipeline for Flux Attention.

Entry point::

    python -m fluxattn.training.train --help

or use ``scripts/train_qwen3_4b.sh``, which wraps the same command with an
FSDP / torchrun launcher.
"""

from .arguments import ScriptArguments, TrainingArguments
from .dataset import PackedDataArguments, build_packed_dataset
from .modeling import PawLlamaForCausalLM, PawQwen3ForCausalLM
from .trainer import Trainer

# For backward compatibility and ease of use
LlamaForCausalLM = PawLlamaForCausalLM
Qwen3ForCausalLM = PawQwen3ForCausalLM

__all__ = [
    "Trainer",
    "ScriptArguments",
    "TrainingArguments",
    "PackedDataArguments",
    "build_packed_dataset",
    "PawLlamaForCausalLM",
    "PawQwen3ForCausalLM",
    # Aliases for backward compatibility
    "LlamaForCausalLM",
    "Qwen3ForCausalLM",
]