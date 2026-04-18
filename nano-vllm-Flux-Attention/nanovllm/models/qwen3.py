import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context
import torch.nn.functional as F


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
            self.cls_router_head_agnostic = nn.Sequential(
                nn.Linear(2 * d_feature, 4 * d_feature),
                nn.SiLU(),
                nn.Linear(4 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 2),
            )
        else:
            self.cls_router_head_agnostic = nn.Sequential(
                nn.Linear(2 * d_feature, 4 * d_feature),
                nn.SiLU(),
                nn.Linear(4 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 1),
            )

        self.tau = temp

    def forward(
        self,
        x,
        cu_seq_len=None,
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


        tau = self.tau

        u = torch.rand_like(binary_logits)
        eps = 1e-8
        g = -torch.log(-torch.log(u + eps) + eps)

        if not self.use_softmax:
            # binary_logits: [B, 1]
            z_soft = torch.sigmoid((binary_logits + g) / tau)  # [B, 1]
            z_hard = (z_soft > 0.5).float()
            z = z_hard

        else:
            # binary_logits: [B, 2]
            z_soft = F.softmax(binary_logits, dim=-1)  # [B, 2]
            z_hard = torch.zeros_like(z_soft).scatter_(
                -1, z_soft.argmax(-1, keepdim=True), 1.0
            )
            z = z_hard

            z = z[..., 1:2]  # [B, 1]

        z_expanded = z.expand(-1, H_dim_size).unsqueeze(-1)

        return {

            "sparse_mask": z_expanded,  # [B, H]

        }

class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()

        self._init_parallel_config(num_heads, num_kv_heads)
        self.head_dim = head_dim or hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self._build_projection(hidden_size)
        self._build_rope(max_position, rope_theta, rope_scaling)
        self._build_attention(rms_norm_eps)

    def _init_parallel_config(self, num_heads, num_kv_heads):
        tp_size = dist.get_world_size()

        assert num_heads % tp_size == 0
        assert num_kv_heads % tp_size == 0

        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads

        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size

    def _build_projection(self, hidden_size):
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=self.qkv_bias,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

    def _build_rope(self, max_position, rope_theta, rope_scaling):
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )

    def _build_attention(self, rms_norm_eps):
        self.mask_allocator = AttentionRouter(
            input_dim=self.head_dim,
            num_key_value_heads=self.total_num_kv_heads,
            d_feature=self.head_dim,
            use_softmax=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def _reshape_qkv(self, q, k, v):
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        return q, k, v

    def _apply_norm(self, q, k):
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k

    def _compute_sparse_gate(self, q, context):
        if not context.is_prefill:
            return 1.0

        router_out = self.mask_allocator(
            x=q,
            cu_seq_len=context.cu_seqlens_q,
        )

        mask_sum = router_out["sparse_mask"].sum()
        gate = float(mask_sum > 0)

        return gate

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        context = get_context()

        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # reshape + norm
        q, k, v = self._reshape_qkv(q, k, v)
        q, k = self._apply_norm(q, k)

        # router gate
        sparse_gate = self._compute_sparse_gate(q, context)

        # rope
        q, k = self.rotary_emb(positions, q, k)

        # attention
        o = self.attn(q, k, v, sparse_gate)

        # output projection
        return self.o_proj(o.flatten(1, -1))


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
