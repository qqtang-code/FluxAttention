# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""

from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

import torch.distributed as dist

import os

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, ModelOutput, LossKwargs
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from flash_attn import flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

from fluxattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
except ImportError:
    raise ImportError(
        "Please install RoPE kernels: `pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary`"
    )
from block_sparse_attn import block_streaming_attn_func

from dataclasses import dataclass

import math

logger = logging.get_logger(__name__)

import torch
import math
import numpy as np
import torch
import torch.nn.functional as F


class PawLlamaConfig(LlamaConfig):
    def __init__(self, *args, **kwargs):
        self.local_window_size = kwargs.pop("local_window_size", 1024)  # 256

        # Streaming
        self.toggle_type = kwargs.pop("toggle_type", "streaming")
        self.sink_size = kwargs.pop("sink_size", 128)

        # retrieval_mode
        self.retrieval_mode = kwargs.pop("retrieval_mode", "full")
        # Head Router
        self.pooling_mode = kwargs.pop("pooling_mode", "first_token")
        self.use_softmax = kwargs.pop("use_softmax", False)
        self.pool_size = kwargs.pop("pool_size", 100)

        # TriangleMix
        self.triangle_n_last = kwargs.pop("triangle_n_last", 128)

        super().__init__(*args, **kwargs)


def rmsnorm_func(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (weight * hidden_states).to(input_dtype)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_buffer(
            "variance_epsilon",
            torch.tensor(eps),
            persistent=False,
        )

    def forward(self, hidden_states):
        return rmsnorm_func(hidden_states, self.weight, self.variance_epsilon)


class FlashRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        scaling_factor=1.0,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        scaling_factor: RotaryEmbedding extended with linear scaling.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
            / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: int = 0,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        if unpadded_lengths is not None:
            cu_seqlens, max_seqlen = unpadded_lengths
        else:
            cu_seqlens, max_seqlen = None, q.shape[1]
        self._update_cos_sin_cache(
            max_seqlen + seqlen_offset, device=q.device, dtype=q.dtype
        )

        if self.scale is None:
            return apply_rotary_emb_func(
                q,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self.interleaved,
                True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ), apply_rotary_emb_func(
                k,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self.interleaved,
                True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            assert False


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        interleaved=False,
        config: Optional[PawLlamaConfig] = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        self.scaling_factor = scaling_factor
        self.interleaved = interleaved
        self.pos_idx_in_fp32 = True

        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"

        self._seq_len_cached = 0

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _update_cos_sin_cache(self, seq_len, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seq_len > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seq_len

            if "dynamic" in self.rope_type:
                inv_freq, self.attention_scaling = self.rope_init_fn(
                    self.config, device, seq_len=seq_len, **self.rope_kwargs
                )
                self.register_buffer("inv_freq", inv_freq, persistent=False)

            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seq_len, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq

            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = (torch.cos(freqs) * self.attention_scaling).to(dtype)
            self._sin_cached = (torch.sin(freqs) * self.attention_scaling).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: int = 0,  # Used in sequence parallelism where each device sees only a chunk of the full sequence
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
    ):
        if unpadded_lengths is not None:
            cu_seqlens, max_seqlen = unpadded_lengths
            if seqlen_offset > 0:
                raise ValueError("seqlen_offset is not supported with unpadded_lengths")
        else:
            cu_seqlens, max_seqlen = None, q.shape[1]

        self._update_cos_sin_cache(max_seqlen + seqlen_offset, q.device, q.dtype)

        return apply_rotary_emb_func(
            q,
            self._cos_cached[seqlen_offset:],
            self._sin_cached[seqlen_offset:],
            self.interleaved,
            True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        ), apply_rotary_emb_func(
            k,
            self._cos_cached[seqlen_offset:],
            self._sin_cached[seqlen_offset:],
            self.interleaved,
            True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@torch.jit.script
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    final_shape = list(hidden_states.shape[:-2]) + [-1] + [hidden_states.shape[-1]]
    expand_shape = [-1] * (len(hidden_states.shape) - 1) + [n_rep] + [-1]
    hidden_states = hidden_states.unsqueeze(-2).expand(expand_shape)
    return hidden_states.reshape(final_shape)


class AttentionRouter(nn.Module):
    def __init__(
        self,
        input_dim,
        num_key_value_heads,
        d_feature=128,
        use_task_emb=False,
        temp=1.0,
        hard=False,
        router_type="mlp",
        use_gumbel=True,
        learnable_temp=False,
        dropout=0.1,
        use_softmax=True,
        pooling_mode="ctx_q",
        pool_size=100,
    ):
        super().__init__()
        self.num_kv = num_key_value_heads
        self.use_task_emb = use_task_emb
        self.router_type = router_type
        self.use_gumbel = use_gumbel
        self.learnable_temp = learnable_temp
        self.pooling_mode = pooling_mode
        self.use_softmax = use_softmax
        self.pool_size = int(pool_size)

        self.cls_feat_extractor = nn.Sequential(
            nn.Linear(d_feature, 8 * d_feature),
            nn.SiLU(),
            nn.Linear(8 * d_feature, 2 * d_feature),
        )

        if self.use_softmax:
            logger.info("using softmax for attention router")
            self.cls_router_head_agnostic = nn.Sequential(
                nn.Linear(2 * d_feature, 4 * d_feature),
                nn.SiLU(),
                nn.Linear(4 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 2),
            )
        else:
            logger.info("use sigmoid function for attention router")
            self.cls_router_head_agnostic = nn.Sequential(
                nn.Linear(2 * d_feature, 4 * d_feature),
                nn.SiLU(),
                nn.Linear(4 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 1),
            )

        if self.use_task_emb:
            self.task_embedding = nn.Embedding(4, d_feature)

        # ---- learnable temperature ----
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temp)))
        else:
            self.tau = temp

    def forward(
        self,
        x,
        cu_seq_len=None,
        range_ids: torch.Tensor = None,
        task_ids: Optional[torch.Tensor] = None,
        current_tau: Optional[torch.Tensor] = None,
    ):
        bsz = (cu_seq_len.shape[0] - 1) if cu_seq_len is not None else 1

        H_dim_size = x.shape[1] if cu_seq_len is not None else x.shape[2]

        if self.pooling_mode == "ctx_q":
            if cu_seq_len is not None:
                B = cu_seq_len.shape[0] - 1
                # x: [Total_Tokens, H, D]
                H_dim_size, D = x.shape[1:]
                sample_features = []
                for i in range(B):
                    x_s, x_e = cu_seq_len[i], cu_seq_len[i + 1]
                    seg_slice = x[x_s:x_e]  # [Ti, H, D]
                    seg_pooled = seg_slice.mean(dim=0)  # [H, D]
                    sample_features.append(seg_pooled)

                pooled_latent = torch.stack(sample_features, dim=0)  # [B, H, D]
            else:
                # [B, T, H, D]
                H_dim_size = x.shape[2]
                target = torch.concat(
                    [x[:, : self.pool_size, :], x[:, -self.pool_size :, :]], dim=1
                ).mean(dim=1)
                # target = x.mean(dim = 1)
                pooled_latent = target  # [B, H, D]
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

        pooled_latent_mean = pooled_latent.mean(dim=1)

        pooled_hidden_states = self.cls_feat_extractor(pooled_latent_mean)

        binary_logits = self.cls_router_head_agnostic(pooled_hidden_states)

        if self.learnable_temp:
            tau = torch.exp(self.log_temp).clamp(0.3, 1.0)
        else:
            tau = current_tau if current_tau is not None else self.tau

        u = torch.rand_like(binary_logits)
        eps = 1e-8
        g = -torch.log(-torch.log(u + eps) + eps)

        if not self.use_softmax:
            # binary_logits: [B, 1]
            z_soft = torch.sigmoid((binary_logits + g) / tau)  # [B, 1]
            z_hard = (z_soft > 0.5).float()
            z = z_hard + (z_soft - z_soft.detach())  # [B, 1]

            entropy = -(
                z_soft * torch.log(z_soft + eps)
                + (1 - z_soft) * torch.log(1 - z_soft + eps)
            ).mean()
        else:
            # binary_logits: [B, 2]
            z_soft = F.softmax(binary_logits, dim=-1)  # [B, 2]
            z_hard = torch.zeros_like(z_soft).scatter_(
                -1, z_soft.argmax(-1, keepdim=True), 1.0
            )
            z = z_hard + (z_soft - z_soft.detach())  # [B, 2]

            z = z[..., 1:2]  # [B, 1]
            z_soft = z_soft[..., 1:2]  # [B, 1]

            entropy = -(z_soft * torch.log(z_soft + eps)).mean()

        # [B, 1] -> [B, H]
        z_soft_expanded = z_soft.expand(-1, H_dim_size)
        z_hard_expanded = (
            z_hard[..., :1].expand(-1, H_dim_size)
            if self.use_softmax
            else z_hard.expand(-1, H_dim_size)
        )
        z_expanded = z.expand(-1, H_dim_size).unsqueeze(-1)

        # Logits: [B, 1] -> [B, H, 1] or [B, 2] -> [B, H, 2]
        if self.use_softmax:
            # binary_logits [B, 2] --> [B, H, 2]
            binary_logits_expanded = binary_logits.unsqueeze(1).expand(
                -1, H_dim_size, -1
            )
            z_hard_full = z_hard  # [B, 2]
            z_hard_return = z_hard_full.unsqueeze(1).expand(
                -1, H_dim_size, -1
            )  # [B, H, 2]
        else:
            binary_logits_expanded = binary_logits.unsqueeze(1).expand(
                -1, H_dim_size, -1
            )
            z_hard_return = z_hard_expanded  # [B, H, 2]

        # Hidden states: [B, D] -> [B, H, D]
        pooled_hidden_states_expanded = pooled_hidden_states.unsqueeze(1).expand(
            -1, H_dim_size, -1
        )

        return {
            "pooled_hidden_states": pooled_hidden_states_expanded,  # [B, H, D]
            "decisions": z_soft_expanded,  # [B, H]
            "hard_decisions": z_hard_return,  # [B, H, 2] (softmax)
            "sparse_mask": z_expanded,  # [B, H]
            "logits": binary_logits_expanded,  # [B, H, 1]
            "entropy": entropy,
        }


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PawLlamaConfig,
        context_window_toggle: Optional[int] = 1024,
        layer_idx: int = 1,
    ):
        """
        @context_window_toggle: if not None, the attention will be limited to a context window specified by this value
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(
                torch.get_default_dtype()
            ),
            persistent=False,
        )

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self._dtype = self.q_proj.weight.dtype

        self.mask_allocator = AttentionRouter(
            input_dim=self.hidden_size,
            num_key_value_heads=self.num_key_value_heads,
            # head_dim = self.head_dim,
            d_feature=self.head_dim,
            use_task_emb=getattr(config, "use_task_emb_for_mask", False),
            temp=getattr(config, "mask_temp", 1.0),
            hard=getattr(config, "mask_hard_sample", False),
            pooling_mode=getattr(config, "pooling_mode", "first_token"),
            use_softmax=getattr(config, "use_softmax", False),
            pool_size=getattr(config, "pool_size", 100),
        )

        self.context_window_toggle = context_window_toggle

        self.sink_blocks = (config.sink_size + 127) // 128
        self.local_blocks = (config.local_window_size + 127) // 128

        self.streaming_info_kwargs = {
            "sink_block_num": self.sink_blocks,
            "local_block_num": self.local_blocks,
        }
        # self.head_indices = self.num_heads // self.num_key_value_heads
        self.head_indices = self.num_heads
        self.granularity = int(getattr(config, "block_size", 64))
        self.xattn_params = {
            "stride": 16,
            "norm": 1,
            "softmax": True,
            "threshold": 0.9,
            "chunk_size": 16384,
            "select_mode": "inverse",
            "use_triton": True,
            "causal": True,
            "kdb": 1,
            "keep_sink": True,
            "keep_recent": True,
        }

        if self.toggle_type == "streaming":
            self.streaming_info_kwargs = {
                "sink_block_num": self.sink_blocks,
                "local_block_num": self.local_blocks,
            }
            self.context_window_toggle = (self.sink_blocks + self.local_blocks) * 128
        elif self.toggle_type == "local":
            pass
        elif self.toggle_type == "triangle":
            self.streaming_info_kwargs = {
                "sink_block_num": self.sink_blocks,
                "local_block_num": self.local_blocks,
            }
            self.context_window_toggle = (self.sink_blocks + self.local_blocks) * 128
            self.triangle_n_last = config.triangle_n_last
        elif self.toggle_type == "xattn" or self.retrieval_mode == "xattn":
            self.streaming_info_kwargs = {
                "sink_block_num": self.sink_blocks,
                "local_block_num": self.local_blocks,
            }
            # self.head_indices = self.num_heads // self.num_key_value_heads
            self.head_indices = self.num_heads
            self.granularity = int(getattr(config, "block_size", 128))
            self.xattn_params = {
                "stride": 16,
                "norm": 1,
                "softmax": True,
                "threshold": 0.9,
                "chunk_size": 16384,
                "select_mode": "inverse",
                "use_triton": True,
                "causal": True,
                "kdb": 1,
                "keep_sink": True,
                "keep_recent": True,
            }
        elif self.toggle_type == "none" or self.toggle_type == "full":
            pass
        else:
            raise ValueError(f"Unknown toggle type: {self.toggle_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        seq_parallel_group: Optional[Any] = None,
        segment_ids: Optional[torch.LongTensor] = None,
        range_ids: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape)
        k = self.k_proj(hidden_states).view(hidden_shape)
        v = self.v_proj(hidden_states).view(hidden_shape)

        has_layer_past = past_key_value is not None

        if not has_layer_past:
            res = self.mask_allocator(q, None, range_ids, task_ids)
            z_kv_batch = res["sparse_mask"]
            if z_kv_batch.shape[-2] == self.num_key_value_heads:
                z_kv_batch = z_kv_batch.repeat_interleave(self.num_key_value_groups, 1)

            sparse_attention_gate = 1 if z_kv_batch.sum(1) != 0 else 0
            past_len = 0
        else:
            past_k = past_key_value[0]
            past_v = past_key_value[1]
            past_len = past_key_value[2]
            z_kv_batch = past_key_value[3]
            sparse_attention_gate = past_key_value[4]

        if position_ids is not None:
            past_len += position_ids.min()

        q, k = self.rotary_emb(q, k, past_len, unpadded_lengths)

        kv = torch.stack([k, v], -3)

        if has_layer_past:
            new_len = past_len + q.size(1)
            if new_len > past_k.size(1):
                pad_k = torch.empty(
                    hidden_states.size(0),
                    256,
                    k.size(2),
                    k.size(3),
                    dtype=k.dtype,
                    device=k.device,
                )
                pad_v = torch.empty(
                    hidden_states.size(0),
                    256,
                    v.size(2),
                    v.size(3),
                    dtype=v.dtype,
                    device=v.device,
                )
                past_k = torch.cat([past_k, pad_k], dim=1)
                past_v = torch.cat([past_v, pad_v], dim=1)

            past_k[:, past_len:new_len] = k
            past_v[:, past_len:new_len] = v
            k_cache = past_k[:, :new_len]
            v_cache = past_v[:, :new_len]
        else:
            past_k = k
            past_v = v
            k_cache = k
            v_cache = v

        past_key_value = (
            (past_k, past_v, past_len + q.size(1), z_kv_batch, sparse_attention_gate)
            if use_cache
            else None
        )

        if not has_layer_past:
            if sparse_attention_gate == 1:
                if self.retrieval_mode == "full":
                    attn_output = flash_attn_func(
                        q,
                        k_cache,
                        v_cache,
                        dropout_p=0.0,
                        softmax_scale=1.0 / self.norm_factor,
                        causal=True,
                    )
                elif self.retrieval_mode == "xattn":
                    k = k.repeat_interleave(self.num_key_value_groups, dim=2)
                    v = v.repeat_interleave(self.num_key_value_groups, dim=2)
                    q, k, v = (
                        q.transpose(1, 2).contiguous(),
                        k.transpose(1, 2).contiguous(),
                        v.transpose(1, 2).contiguous(),
                    )
                    bsz, _, seqlen, _ = q.size()
                    max_seqlen = seqlen

                    cu_seqlens = torch.arange(
                        0,
                        (bsz + 1) * seqlen,
                        step=seqlen,
                        dtype=torch.int32,
                        device=q.device,
                    )
                    unpadded_lengths_xattn = (cu_seqlens, max_seqlen)

                    cu_seqlens, max_seqlen = unpadded_lengths_xattn
                    stride = self.xattn_params["stride"]
                    threshold = self.xattn_params["threshold"]
                    norm = self.xattn_params["norm"]
                    attn_output = Xattention_prefill_dim4(
                        q,
                        k,
                        v,
                        stride,
                        cu_seqlens,
                        norm,
                        threshold,
                        use_triton=True,
                    ).transpose(1, 2)  # B, T, H, D
                elif self.retrieval_mode == "triangle":
                    k_cache = k_cache.repeat_interleave(
                        self.num_key_value_groups, dim=2
                    )
                    v_cache = v_cache.repeat_interleave(
                        self.num_key_value_groups, dim=2
                    )
                    n_last = self.triangle_n_last
                    n_last = min(n_last, q.size(1) - 1)
                    q1, q2 = q[:, :-n_last, :, :], q[:, -n_last:, :, :]

                    y1 = streaming_attn_func(
                        q1,
                        k_cache[:, :-n_last, :, :],
                        v_cache[:, :-n_last, :, :],
                        self.streaming_info_kwargs,
                        dropout_p=0.0,
                        causal=True,
                        return_attn_probs=False,
                    )

                    k_cache = k_cache.transpose(1, 2)
                    v_cache = v_cache.transpose(1, 2)
                    q2 = q2.transpose(1, 2)
                    scale = 1.0 / math.sqrt(q2.shape[-1])
                    qk = torch.einsum("bhmk, bhnk -> bhmn", q2, k_cache) * scale

                    arange = torch.arange(n_last, device=q.device)
                    mask = arange[None, :] > arange[:, None]
                    mask_section = qk[:, :, :, -n_last:]
                    qk[:, :, :, -n_last:] = mask_section.masked_fill(
                        mask, float("-inf")
                    )
                    attn_weights = torch.nn.functional.softmax(
                        qk, dim=-1, dtype=torch.float32
                    ).to(q.dtype)
                    y2 = torch.einsum("bhmn, bhnk -> bhmk", attn_weights, v_cache)

                    y2 = y2.transpose(1, 2)

                    attn_output = torch.cat([y1, y2], dim=1)
            else:
                if self.toggle_type == "streaming":
                    bsz, seqlen = q.size(0), q.size(1)
                    cu_seqlens_q = torch.arange(
                        0,
                        (bsz + 1) * seqlen,
                        step=seqlen,
                        dtype=torch.int32,
                        device=q.device,
                    )
                    cu_seqlens_k = cu_seqlens_q

                    head_mask_type = torch.full(
                        (self.num_heads,), -1, device=q.device, dtype=torch.int32
                    )
                    streaming_info = torch.tensor(
                        [self.sink_blocks, self.local_blocks] * self.num_heads,
                        device=q.device,
                        dtype=torch.int32,
                    )

                    attn_output = block_streaming_attn_func(
                        q.view(-1, self.num_heads, self.head_dim),
                        k_cache.view(-1, self.num_key_value_heads, self.head_dim),
                        v_cache.view(-1, self.num_key_value_heads, self.head_dim),
                        cu_seqlens_q,
                        cu_seqlens_k,
                        head_mask_type,
                        streaming_info,
                        max_seqlen_q_=seqlen,
                        max_seqlen_k_=seqlen,
                        p_dropout=0.0,
                        deterministic=False,
                        softmax_scale=None,
                        is_causal=True,
                        return_attn_probs=False,
                    ).view(bsz, seqlen, self.num_heads, self.head_dim)
                elif self.toggle_type == "xattn":
                    k = k.repeat_interleave(self.num_key_value_groups, dim=2)
                    v = v.repeat_interleave(self.num_key_value_groups, dim=2)
                    q, k, v = (
                        q.transpose(1, 2).contiguous(),
                        k.transpose(1, 2).contiguous(),
                        v.transpose(1, 2).contiguous(),
                    )
                    bsz, _, seqlen, _ = q.size()
                    max_seqlen = seqlen

                    cu_seqlens = torch.arange(
                        0,
                        (bsz + 1) * seqlen,
                        step=seqlen,
                        dtype=torch.int32,
                        device=q.device,
                    )
                    unpadded_lengths_xattn = (cu_seqlens, max_seqlen)

                    cu_seqlens, max_seqlen = unpadded_lengths_xattn
                    stride = self.xattn_params["stride"]
                    threshold = self.xattn_params["threshold"]
                    norm = self.xattn_params["norm"]
                    attn_output = Xattention_prefill_dim4(
                        q,
                        k,
                        v,
                        stride,
                        cu_seqlens,
                        norm,
                        threshold,
                        use_triton=True,
                    ).transpose(1, 2)  # B, T, H, D
                elif self.toggle_type == "triangle":
                    k_cache = k_cache.repeat_interleave(
                        self.num_key_value_groups, dim=2
                    )
                    v_cache = v_cache.repeat_interleave(
                        self.num_key_value_groups, dim=2
                    )
                    n_last = self.triangle_n_last
                    n_last = min(n_last, q.size(1) - 1)
                    q1, q2 = q[:, :-n_last, :, :], q[:, -n_last:, :, :]

                    y1 = streaming_attn_func(
                        q1,
                        k_cache[:, :-n_last, :, :],
                        v_cache[:, :-n_last, :, :],
                        self.streaming_info_kwargs,
                        dropout_p=0.0,
                        causal=True,
                        return_attn_probs=False,
                    )

                    k_cache = k_cache.transpose(1, 2)
                    v_cache = v_cache.transpose(1, 2)
                    q2 = q2.transpose(1, 2)
                    scale = 1.0 / math.sqrt(q2.shape[-1])
                    qk = torch.einsum("bhmk, bhnk -> bhmn", q2, k_cache) * scale

                    arange = torch.arange(n_last, device=q.device)
                    mask = arange[None, :] > arange[:, None]
                    mask_section = qk[:, :, :, -n_last:]
                    qk[:, :, :, -n_last:] = mask_section.masked_fill(
                        mask, float("-inf")
                    )
                    attn_weights = torch.nn.functional.softmax(
                        qk, dim=-1, dtype=torch.float32
                    ).to(q.dtype)
                    y2 = torch.einsum("bhmn, bhnk -> bhmk", attn_weights, v_cache)

                    y2 = y2.transpose(1, 2)

                    attn_output = torch.cat([y1, y2], dim=1)
        else:
            if sparse_attention_gate == 1:
                bsz = q.size(0)
                seqlen_int = k_cache.size(1)
                cache_seqlens = torch.full(
                    (bsz,), seqlen_int, dtype=torch.int32, device=q.device
                )

                attn_output = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_seqlens=cache_seqlens,
                    softmax_scale=1.0 / self.norm_factor,
                    causal=True,
                )
            else:
                bsz, seqlen, _, head_dim = k_cache.size()

                if (
                    getattr(self, "_streaming_info", None) is None
                    or self._streaming_info.device != q.device
                ):
                    self._cu_seqlens_q = torch.tensor(
                        [0, 1], dtype=torch.int32, device=q.device
                    )
                    self._head_mask_type = torch.full(
                        (self.num_heads,), -1, device=q.device, dtype=torch.int32
                    )
                    self._streaming_info = torch.tensor(
                        [self.sink_blocks, self.local_blocks] * self.num_heads,
                        device=q.device,
                        dtype=torch.int32,
                    )

                cu_seqlens_k = torch.tensor(
                    [0, seqlen], dtype=torch.int32, device=q.device
                )

                attn_output = (
                    block_streaming_attn_func(
                        q.squeeze(0),
                        k_cache.squeeze(0),
                        v_cache.squeeze(0),
                        self._cu_seqlens_q,
                        cu_seqlens_k,
                        self._head_mask_type,
                        self._streaming_info,
                        max_seqlen_q_=1,
                        max_seqlen_k_=seqlen,
                        p_dropout=0.0,
                        deterministic=False,
                        softmax_scale=None,
                        is_causal=True,
                        return_attn_probs=False,
                    )
                    .unsqueeze(0)
                    .contiguous()
                )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output.to(self.o_proj.weight.dtype))

        attn_weights = None
        return (
            z_kv_batch.squeeze(-1).sum(dim=-1),
            attn_output,
            attn_weights,
            past_key_value,
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PawLlamaConfig,
        context_window_toggle: Optional[int] = 4096,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(
            config=config,
            context_window_toggle=context_window_toggle,
            layer_idx=layer_idx,
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._fsdp_wrap = True

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        self.self_attn.set_threshold_for_deterministic(threshold_for_deterministic)

    @torch.no_grad()
    def get_masks(self):
        return self.self_attn.get_masks()

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        self.self_attn.reset_masks(value)

    @torch.no_grad()
    def fill_masks_with_value(self, value):
        self.self_attn.fill_masks_with_value(value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        seq_parallel_group: Optional[Any] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        (
            z_sum,
            hidden_states,
            self_attn_weights,
            present_key_value,
        ) = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            unpadded_lengths=unpadded_lengths,
            seq_parallel_group=seq_parallel_group,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (
            z_sum,
            hidden_states,
        )

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = PawLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@dataclass
class BaseModelOutputWithPastAndSparsity(ModelOutput):
    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    model_sparsity: Optional[torch.FloatTensor] = None


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: PawLlamaConfig
    """

    def __init__(
        self,
        config: PawLlamaConfig,
    ):
        super().__init__(config)
        context_window_toggle = config.local_window_size

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config, context_window_toggle=context_window_toggle, layer_idx=i
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.total_num_heads = config.num_attention_heads * config.num_hidden_layers
        self.total_num_kv_heads = config.num_key_value_heads * config.num_hidden_layers

        self._dtype = self.norm.weight.dtype

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def get_sparsity(self):
        masks = self.get_masks()
        total_sum = 0
        for mask in masks:
            total_sum += mask.sum()
        return 1 - (total_sum / self.total_num_kv_heads)

    @torch.no_grad()
    def _pre_save_get_threshold(self):
        orig_threshold = self.threshold_for_deterministic

        sparsity_target = self.get_sparsity()
        l = 0
        r = 1
        while r - l > 1e-8:
            m = (l + r) / 2
            self.set_threshold_for_deterministic(m)
            if self.get_sparsity() > sparsity_target:
                r = m
            else:
                l = m
        m = (l + r) / 2

        self.config.suggested_threshold = m

    @torch.no_grad()
    def reset_masks_with_stripe_pattern(self, width_1, width_2, start_with_keep=True):
        if start_with_keep:
            value_1 = 10.0  # Some high value
            value_2 = -10.0  # Some low value
        else:
            value_1 = -10.0
            value_2 = 10.0
        for l, layer in range(len(self.layers)):
            value = value_1 if l % (width_1 + width_2) < width_1 else value_2
            layer.fill_masks_with_value(value)

    @torch.no_grad()
    def load_masks(self, masks):
        for l in range(len(masks)):
            self.layers[l].fill_masks_with_value(masks[l])

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        seq_parallel_group: Optional[Any] = None,
        # enable_contrastive_loss: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        compute_sparsity = self.training
        # compute_sparsity = True
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # position_ids = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        z_sum = 0 if compute_sparsity else None
        layer_z_sums = []

        head_entropy = 0 if compute_sparsity else None
        layer_z_constrast = []

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if past_key_values is not None and len(past_key_values) > idx:
                past_key_value = past_key_values[idx]
            else:
                past_key_value = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    unpadded_lengths,
                    output_attentions,
                    False,
                    seq_parallel_group,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    unpadded_lengths=unpadded_lengths,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    seq_parallel_group=seq_parallel_group,
                )

            z_layer_sum, hidden_states = (
                layer_outputs[0],
                layer_outputs[1],
            )

            z_layer_sum = z_layer_sum.to(hidden_states.device)

            if z_sum is None:
                z_sum = z_layer_sum
            else:
                z_sum = z_sum.to(z_layer_sum.device)
                z_sum = z_sum + z_layer_sum

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 2],)

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        model_sparsity = 1 - (z_sum / self.total_num_heads)

        if not return_dict:
            # return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, model_sparsity, target_sparsity, z_loss] if v is not None)
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    model_sparsity,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndSparsity(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            model_sparsity=model_sparsity,
        )


@dataclass
class CausalLMOutputWithPastAndSparsity(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    model_sparsity: Optional[torch.FloatTensor] = None


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class PawLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.model = LlamaModel(
            config,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.logit_block_size = int(os.environ.get("LOGIT_BLOCK_SIZE", 16384))
        self.prefill_sparsity = None
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def compute_loss(self, hidden_states, labels):
        if (labels != -100).sum() == 0:
            return torch.tensor(
                0.0, device=hidden_states.device, dtype=hidden_states.dtype
            )
        min_len = min(hidden_states.size(0), labels.size(0))
        hidden_states = hidden_states[:min_len]
        labels = labels[:min_len]

        logits = self.lm_head(hidden_states)
        if len(logits.shape) > 2:
            logits = logits.transpose(-1, -2)
        return F.cross_entropy(
            logits,
            labels,
            ignore_index=-100,
            reduction=("sum" if getattr(self, "token_scaled_loss", False) else "mean"),
        )

    def save_pretrained(self, *args, **kwargs):
        # First save the suggested threshold
        self.model._pre_save_get_threshold()
        return super().save_pretrained(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        return_token_losses: bool = False,
        shifted_labels: Optional[torch.LongTensor] = None,
        seq_parallel_group: Optional[Any] = None,
        target_sparsity: Optional[float] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if seq_lengths is not None:
            if inputs_embeds is not None:
                assert len(inputs_embeds.shape) == 2, (
                    "inputs_embeds should be a 2D tensor with `seq_lengths`"
                )
                # assert inputs_embeds.size(0) == seq_lengths.sum(), "inputs_embeds and seq_lengths should have the same batch size"
            else:
                assert len(input_ids.shape) == 1, (
                    "input_ids should be a 1D tensor with `seq_lengths`"
                )
                # assert input_ids.size(0) == seq_lengths.sum(), "input_ids and seq_lengths should have the same batch size"

            assert attention_mask is None or attention_mask.all().item(), (
                "attention_mask should be None or all ones for `seq_lengths`"
            )
            assert not use_cache, "use_cache is not supported with `seq_lengths`"

            cu_seqlens = F.pad(
                torch.cumsum(seq_lengths, dim=0, dtype=torch.torch.int32), (1, 0)
            )
            max_seqlen = seq_lengths.max().item()

            unpadded_lengths = (cu_seqlens, max_seqlen)
        elif (
            attention_mask is not None and not use_cache and attention_mask.size(0) != 1
        ):
            if inputs_embeds is not None:
                bsz = inputs_embeds.size(0)
                inputs_embeds, unpad_indices, cu_seqlens, max_seqlen = unpad_input(
                    inputs_embeds, attention_mask
                )
            else:
                bsz = input_ids.size(0)
                tmp = input_ids.unsqueeze(-1)
                input_ids, unpad_indices, cu_seqlens, max_seqlen = unpad_input(
                    tmp, attention_mask
                )
                max_seqlen_for_pad_seq = attention_mask.size(-1)
                input_ids = input_ids.squeeze(-1)
            unpadded_lengths = (cu_seqlens, max_seqlen)
        else:
            unpadded_lengths = None
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            unpadded_lengths=unpadded_lengths,
            seq_parallel_group=seq_parallel_group,
        )

        if input_ids.shape[1] > 1 and use_cache:
            self.prefill_sparsity = outputs.model_sparsity.detach()

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -1:, :])
        loss = None
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndSparsity(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            model_sparsity=outputs.model_sparsity,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # These are static or need special handling during generation
        custom_keys = ["segment_ids", "range_ids", "task_ids"]
        for key in custom_keys:
            if key in kwargs:
                value = kwargs[key]
                # For segment_ids: may need to extend to match input_ids length (if generating)
                if key == "segment_ids" and value is not None:
                    # Extend segment_ids with answer segment ID (3) for new tokens
                    if value.shape[1] < input_ids.shape[1]:
                        pad_len = input_ids.shape[1] - value.shape[1]
                        pad_seg = torch.full(
                            (value.shape[0], pad_len),
                            fill_value=3,  # answer segment ID (as in training)
                            dtype=value.dtype,
                            device=value.device,
                        )
                        value = torch.cat([value, pad_seg], dim=1)
                model_inputs[key] = value

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


if __name__ == "__main__":
    from transformers import AutoTokenizer, LlamaForCausalLM

    path = ""
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(path)

    model = LlamaForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    prompt = "你好，你是谁?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # input = torch.tensor([[100,2000,33004,4420]], device="cuda")
    output = model.generate(inputs=inputs["input_ids"])
    print(output)

    output_ids = output[0]

    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(text)
