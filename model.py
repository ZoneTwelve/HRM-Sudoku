# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class HRMACTModelConfig:
    @dataclass
    class TransformerConfig:
        num_layers: int
        hidden_size: int
        num_heads: int
        expansion: float = 4.0
        norm_epsilon: float = 1e-5
        rope_theta: float = 10000.0

    @dataclass
    class ACTConfig:
        halt_max_steps: int
        halt_exploration_probability: float

    seq_len: int
    vocab_size: int
    high_level_cycles: int
    low_level_cycles: int
    transformers: TransformerConfig
    act: ACTConfig
    dtype: torch.dtype = torch.bfloat16

class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.epsilon = eps

    def forward(self, x):
        # Cast to float32 for stability, as in the original implementation
        original_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.epsilon)).to(original_dtype)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_length: int, base: float, dtype: torch.dtype):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
        t = torch.arange(max_length, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(-2)

        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        return (x * self.cos) + (self._rotate_half(x) * self.sin)

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.qkv_proj = nn.Linear(dim, (num_heads * 2 + num_heads) * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

    def forward(self, x, rope):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, L, self.num_heads * 3, self.head_dim)

        query, key, value = qkv.split([self.num_heads, self.num_heads, self.num_heads], dim=2)

        query = rope(query)
        key = rope(key)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(query, key, value)

        output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(output)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, expansion: float):
        super().__init__()
        hidden_dim = int(expansion * dim * 2.0 / 3.0)
        hidden_dim = (-(hidden_dim // -256)) * 256

        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

class HRMACTBlock(nn.Module):
    def __init__(self, config: HRMACTModelConfig.TransformerConfig):
        super().__init__()
        self.self_attn = Attention(
            dim=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.hidden_size // config.num_heads
        )
        self.mlp = SwiGLU(dim=config.hidden_size, expansion=config.expansion)
        self.norm1 = RMSNorm(eps=config.norm_epsilon)
        self.norm2 = RMSNorm(eps=config.norm_epsilon)

    def forward(self, x, rope):
        # FIX: Apply Post-LayerNorm, matching the original Swift implementation
        # The input 'x' is passed directly to the sub-layers.
        x = self.norm1(x + self.self_attn(x, rope))
        x = self.norm2(x + self.mlp(x))
        return x

class HRMACTReasoner(nn.Module):
    def __init__(self, config: HRMACTModelConfig.TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([HRMACTBlock(config) for _ in range(config.num_layers)])

    def forward(self, hidden_state, input_injection, rope):
        hidden_state = hidden_state + input_injection
        for layer in self.layers:
            hidden_state = layer(hidden_state, rope)
        return hidden_state

class HRMACTInner(nn.Module):
    def __init__(self, config: HRMACTModelConfig):
        super().__init__()
        self.config = config

        self.cls_token = nn.Parameter(torch.empty(config.transformers.hidden_size))

        self.input_embedding = nn.Embedding(config.vocab_size, config.transformers.hidden_size)

        self.output_head = nn.Linear(config.transformers.hidden_size, config.vocab_size, bias=False)

        self.q_act_head = nn.Linear(config.transformers.hidden_size, 2)

        self.rotary_emb = RotaryPositionEmbedding(
            dim=config.transformers.hidden_size // config.transformers.num_heads,
            max_length=config.seq_len + 1,
            base=config.transformers.rope_theta,
            dtype=config.dtype
        )

        self.high_level_reasoner = HRMACTReasoner(config.transformers)
        self.low_level_reasoner = HRMACTReasoner(config.transformers)

        self.initial_high_level = nn.Parameter(torch.empty(config.transformers.hidden_size))
        self.initial_low_level = nn.Parameter(torch.empty(config.transformers.hidden_size))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
             torch.nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)

        if hasattr(self, 'initial_high_level'):
            torch.nn.init.trunc_normal_(self.initial_high_level, std=1.0, a=-2.0, b=2.0)
            torch.nn.init.trunc_normal_(self.initial_low_level, std=1.0, a=-2.0, b=2.0)
            if self.cls_token.numel() > 0: # Check if parameter is initialized
                torch.nn.init.trunc_normal_(self.cls_token, std=1.0 / math.sqrt(self.config.transformers.hidden_size))

        if hasattr(self, 'q_act_head'):
            if self.q_act_head.weight.numel() > 0: # Check if parameter is initialized
                nn.init.zeros_(self.q_act_head.weight)
                # FIX: Initialize bias to 0.0 to encourage exploration
                nn.init.zeros_(self.q_act_head.bias)


    def forward(self, hidden_states, inputs):
        low_level_z, high_level_z = hidden_states
        batch_size = inputs.shape[0]

        input_embeddings = self.input_embedding(inputs)

        cls_tokens = self.cls_token.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

        full_embeddings = torch.cat([cls_tokens, input_embeddings], dim=1)
        full_embeddings *= math.sqrt(self.config.transformers.hidden_size)

        total_cycles = self.config.high_level_cycles * self.config.low_level_cycles

        for cycle in range(1, total_cycles):
            low_level_z = self.low_level_reasoner(
                hidden_state=low_level_z,
                input_injection=high_level_z + full_embeddings,
                rope=self.rotary_emb
            )
            if cycle % self.config.low_level_cycles == 0:
                high_level_z = self.high_level_reasoner(
                    hidden_state=high_level_z,
                    input_injection=low_level_z,
                    rope=self.rotary_emb
                )

        low_level_z = low_level_z.detach()
        high_level_z = high_level_z.detach()

        low_level_z = self.low_level_reasoner(
            hidden_state=low_level_z,
            input_injection=high_level_z + full_embeddings,
            rope=self.rotary_emb
        )
        high_level_z = self.high_level_reasoner(
            hidden_state=high_level_z,
            input_injection=low_level_z,
            rope=self.rotary_emb
        )

        output_logits = self.output_head(high_level_z[:, 1:])
        q_act_logits = self.q_act_head(high_level_z[:, 0])

        q_act_halt = q_act_logits[:, 0]
        q_act_continue = q_act_logits[:, 1]

        new_hidden_states = (low_level_z.detach(), high_level_z.detach())

        return new_hidden_states, output_logits, q_act_halt, q_act_continue
