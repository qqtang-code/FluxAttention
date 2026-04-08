from .lh_trainer import Trainer
from .modeling_flash_llama import PawLlamaForCausalLM
from .modeling_flash_qwen import PawQwen3ForCausalLM
from .script_arguments import ScriptArguments

# For backward compatibility and ease of use
LlamaForCausalLM = PawLlamaForCausalLM
Qwen3ForCausalLM = PawQwen3ForCausalLM

__all__ = [
    "Trainer",
    "PawLlamaForCausalLM",
    "PawQwen3ForCausalLM",
    "ScriptArguments",
    # Aliases for backward compatibility
    "LlamaForCausalLM",
    "Qwen3ForCausalLM",
]
