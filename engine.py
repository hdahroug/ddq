"""engine.py — standalone Qwen2 inference engine (no HuggingFace / transformers).

Owns the full inference path with hand-written CUDA kernels. The tokenizer is
intentionally out of scope: callers supply token ids.

Kernels used (all in this directory):
    RMSnorm_d.cu, RoPe_d.cu, SwiGLU_d.cu, gemm_d.cu, sampler_d.cu, FA_d.cu

Attention routing (Phase 1):
    prefill  Hopper -> FA_d.cu        non-Hopper -> torch SDPA
    decode   all    -> torch SDPA     (FAD_d.cu / FlashInfer land in later phases)
"""

import json
import os

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file
from torch.utils.cpp_extension import load as _kernel_load

_DIR = os.path.dirname(os.path.abspath(__file__))

rmsnorm_naive = _kernel_load(
    name="engine_rmsnorm",
    sources=[os.path.join(_DIR, "RMSnorm_d.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
rope_naive = _kernel_load(
    name="engine_rope",
    sources=[os.path.join(_DIR, "RoPe_d.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
swiglu_naive = _kernel_load(
    name="engine_swiglu",
    sources=[os.path.join(_DIR, "SwiGLU_d.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
sampler_naive = _kernel_load(
    name="engine_sampler",
    sources=[os.path.join(_DIR, "sampler_d.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
gemm_naive = _kernel_load(
    name="engine_gemm",
    sources=[os.path.join(_DIR, "gemm_d.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

_capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
_HOPPER = _capability[0] == 9
_fa_cflags = ["-O3", "--use_fast_math"]
_fa_ldflags = []
if _HOPPER:
    _fa_cflags += [
        "--std=c++20",
        "-DENABLE_HOPPER_TMA_WGMMA=1",
        "-gencode=arch=compute_90a,code=sm_90a",
    ]
    _fa_ldflags = ["-lcuda"]
fa_naive = _kernel_load(
    name="engine_fa",
    sources=[os.path.join(_DIR, "FA_d.cu")],
    extra_cuda_cflags=_fa_cflags,
    extra_ldflags=_fa_ldflags,
    verbose=False,
)
fad_naive = _kernel_load(
    name="engine_fad",
    sources=[os.path.join(_DIR, "FAD_d.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++20"],
    verbose=False,
)

# Phase 4: non-Hopper fallback routes here instead of torch SDPA.
_flashinfer = None


def _linear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return gemm_naive.forward(x, module.weight, module.bias)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _build_causal_mask(q_len: int, kv_len: int, past: int, device) -> torch.Tensor:
    q_pos = torch.arange(past, past + q_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    return k_pos <= q_pos


class KVCache:
    """Preallocated per-layer KV cache. Grows geometrically; never per-token cat."""

    def __init__(self, num_layers, batch, num_kv_heads, head_dim, device, dtype, capacity=256):
        shape = (num_layers, batch, num_kv_heads, capacity, head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.seq_len = 0

    def _grow(self, needed: int) -> None:
        capacity = self.k.size(3)
        if needed <= capacity:
            return
        new_capacity = capacity
        while new_capacity < needed:
            new_capacity *= 2
        shape = (*self.k.shape[:3], new_capacity, self.k.shape[4])
        new_k = torch.empty(shape, device=self.k.device, dtype=self.k.dtype)
        new_v = torch.empty(shape, device=self.v.device, dtype=self.v.dtype)
        new_k[:, :, :, :capacity, :] = self.k
        new_v[:, :, :, :capacity, :] = self.v
        self.k, self.v = new_k, new_v

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start: int) -> None:
        length = k.size(2)
        self._grow(start + length)
        self.k[layer_idx, :, :, start:start + length, :] = k
        self.v[layer_idx, :, :, start:start + length, :] = v

    def view(self, layer_idx: int, end: int):
        return self.k[layer_idx, :, :, :end, :], self.v[layer_idx, :, :, :end, :]


class _Config:
    def __init__(self, raw: dict):
        self.hidden_size = raw["hidden_size"]
        self.intermediate_size = raw["intermediate_size"]
        self.num_hidden_layers = raw["num_hidden_layers"]
        self.num_attention_heads = raw["num_attention_heads"]
        self.num_key_value_heads = raw["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_kv_groups = self.num_attention_heads // self.num_key_value_heads
        self.rms_norm_eps = raw["rms_norm_eps"]
        self.rope_theta = float(raw["rope_theta"])
        self.max_position_embeddings = raw["max_position_embeddings"]
        self.vocab_size = raw["vocab_size"]
        self.pad_token_id = raw.get("pad_token_id")


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return rmsnorm_naive.forward(x.contiguous(), self.weight, self.eps)


class _RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        self._table_ready = False

    def _build_table(self, device):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim))
        cos, sin = rope_naive.init_table(inv_freq.to(device).contiguous(), self.max_seq_len)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)
        self._table_ready = True

    def forward(self, position_ids):
        return self.cos_table[position_ids], self.sin_table[position_ids]


class _MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate = _linear(self.gate_proj, x)
        up = _linear(self.up_proj, x)
        z = swiglu_naive.forward(gate.view(-1, self.intermediate_size), up.view(-1, self.intermediate_size))
        return _linear(self.down_proj, z.view(gate.shape))


class _Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_kv_groups
        self.scaling = self.head_dim ** -0.5
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

    def forward(self, hidden_states, cos, sin, cache, start, end):
        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)

        query = _linear(self.q_proj, hidden_states).view(hidden_shape).transpose(1, 2)
        key = _linear(self.k_proj, hidden_states).view(hidden_shape).transpose(1, 2)
        value = _linear(self.v_proj, hidden_states).view(hidden_shape).transpose(1, 2)

        query = rope_naive.forward(query.contiguous(), cos, sin)
        key = rope_naive.forward(key.contiguous(), cos, sin)

        cache.write(self.layer_idx, key, value, start)
        key, value = cache.view(self.layer_idx, end)

        if _HOPPER and seq_len == 1:
            attn_output = fad_naive.forward(
                query.contiguous(), key.contiguous(), value.contiguous(), float(self.scaling)
            )
            attn_output = attn_output.transpose(1, 2)
        else:
            key = repeat_kv(key, self.num_kv_groups)
            value = repeat_kv(value, self.num_kv_groups)

            full_prefill = start == 0 and seq_len == end
            if seq_len == 1:
                attn_mask, is_causal = None, False
            elif full_prefill:
                attn_mask, is_causal = None, True
            else:
                attn_mask, is_causal = _build_causal_mask(seq_len, end, start, query.device), False

            if _HOPPER and seq_len > 1 and full_prefill:
                attn_output = fa_naive.forward(
                    query.contiguous(), key.contiguous(), value.contiguous(), True, float(self.scaling)
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attn_mask, dropout_p=0.0,
                    scale=self.scaling, is_causal=is_causal,
                )
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(batch, seq_len, -1)
        return _linear(self.o_proj, attn_output)


class _DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = _Attention(config, layer_idx)
        self.mlp = _MLP(config)
        self.input_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, cache, start, end):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), cos, sin, cache, start, end)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class _Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([_DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary = _RotaryEmbedding(config.head_dim, config.rope_theta, config.max_position_embeddings)

    def forward(self, input_ids, cache):
        start = cache.seq_len
        seq_len = input_ids.shape[1]
        end = start + seq_len

        hidden_states = self.embed_tokens(input_ids)
        if not self.rotary._table_ready:
            self.rotary._build_table(hidden_states.device)
        position_ids = torch.arange(start, end, device=input_ids.device)
        cos, sin = self.rotary(position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, cache, start, end)

        cache.seq_len = end
        return self.norm(hidden_states)


class EngineOutput:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class Engine(nn.Module):
    def __init__(self, config: _Config):
        super().__init__()
        self.config = config
        self.model = _Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def load(cls, model_dir, device="cuda", dtype=torch.float16, batch_size=1):
        with open(os.path.join(model_dir, "config.json")) as handle:
            config = _Config(json.load(handle))
        engine = cls(config)
        weights = load_file(os.path.join(model_dir, "model.safetensors"), device="cpu")
        missing, unexpected = engine.load_state_dict(weights, strict=False)
        if unexpected or set(missing) - {"lm_head.weight"}:
            raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
        engine.lm_head.weight = engine.model.embed_tokens.weight
        engine._batch_size = batch_size
        return engine.to(dtype).to(device).eval()

    def new_cache(self, batch_size=None):
        batch_size = batch_size or getattr(self, "_batch_size", 1)
        return KVCache(
            self.config.num_hidden_layers, batch_size, self.config.num_key_value_heads,
            self.config.head_dim, self.lm_head.weight.device, self.lm_head.weight.dtype,
        )

    def forward(self, input_ids, past_key_values=None):
        if past_key_values is None:
            past_key_values = self.new_cache(batch_size=input_ids.shape[0])
        hidden_states = self.model(input_ids, past_key_values)
        logits = _linear(self.lm_head, hidden_states)
        return EngineOutput(logits, past_key_values)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=30, eos_token_id=None):
        output = self.forward(input_ids)
        cache = output.past_key_values
        next_token = sampler_naive.forward(output.logits[:, -1:, :].contiguous())
        tokens = [next_token.item()]
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and tokens[-1] == eos_token_id:
                break
            output = self.forward(next_token, cache)
            next_token = sampler_naive.forward(output.logits[:, -1:, :].contiguous())
            tokens.append(next_token.item())
        return tokens


__all__ = ["Engine", "EngineOutput", "KVCache"]
