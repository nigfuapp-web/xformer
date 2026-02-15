# Xoron Model Implementation for kt-kernel
# SPDX-License-Identifier: Apache-2.0

"""
Xoron Multimodal Model Implementation for kt-kernel.

This module provides the main XoronForCausalLM class that integrates with
kt-kernel for high-performance CPU/GPU inference.
"""

import os
import math
import logging
from typing import Optional, Dict, List, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration import XoronConfig

logger = logging.getLogger(__name__)

# FP16 safe max value for hidden states
MAX_HIDDEN = 10000.0


def safe_clamp_tensor(x: torch.Tensor, max_val: float = MAX_HIDDEN) -> torch.Tensor:
    """Clamp tensor values for FP16 safety."""
    if x is None or x.numel() == 0:
        return x
    x = torch.nan_to_num(x, nan=0.0, posinf=max_val, neginf=-max_val)
    return x.clamp(-max_val, max_val)


@dataclass
class XoronModelOutput:
    """Output class for Xoron model."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[List[Any]] = None
    aux_loss: Optional[torch.Tensor] = None
    # Multimodal outputs
    image_features: Optional[torch.Tensor] = None
    video_features: Optional[torch.Tensor] = None
    audio_features: Optional[torch.Tensor] = None
    generated_images: Optional[torch.Tensor] = None
    generated_videos: Optional[torch.Tensor] = None
    generated_audio: Optional[torch.Tensor] = None


class YaRNRotaryEmbedding(nn.Module):
    """YaRN (Yet another RoPE extensioN) with LongRoPE-style improvements."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 500000.0,
        original_max_position_embeddings: int = 8192,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.original_max_position = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        
        self.scaling_factor = max_position_embeddings / original_max_position_embeddings
        
        inv_freq = self._compute_yarn_inv_freq()
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def _compute_yarn_inv_freq(self) -> torch.Tensor:
        """Compute YaRN-scaled inverse frequencies."""
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scaling_factor * pos_freqs)
        
        low = max(math.floor(self.dim * math.log(self.original_max_position / (self.beta_fast * 2 * math.pi)) / 
                            (2 * math.log(self.base))), 0)
        high = min(math.ceil(self.dim * math.log(self.original_max_position / (self.beta_slow * 2 * math.pi)) /
                            (2 * math.log(self.base))), self.dim - 1)
        
        inv_freq = torch.zeros(self.dim // 2, dtype=torch.float32)
        for i in range(self.dim // 2):
            if i < low:
                inv_freq[i] = inv_freq_interpolation[i]
            elif i > high:
                inv_freq[i] = inv_freq_extrapolation[i]
            else:
                smooth = (i - low) / max(high - low, 1)
                inv_freq[i] = (1 - smooth) * inv_freq_interpolation[i] + smooth * inv_freq_extrapolation[i]
        
        return inv_freq

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        inv_freq = self.inv_freq.to(device)
        
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        def get_mscale(scale: float) -> float:
            if scale <= 1:
                return 1.0
            return 0.1 * math.log(scale) + 1.0
        
        mscale = get_mscale(self.scaling_factor) * self.mscale
        
        cos = emb.cos().to(dtype=x.dtype) * mscale
        sin = emb.sin().to(dtype=x.dtype) * mscale
        
        return cos, sin


class RMSNorm(nn.Module):
    """RMS Normalization layer."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) for compressed KV cache.
    
    This implements the MLA mechanism from DeepSeek-V2/V3 that compresses
    key-value pairs to reduce memory usage while maintaining quality.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int = None,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 0,
        rope_dim: int = None,
        max_position_embeddings: int = 131072,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_dim = rope_dim or self.head_dim
        
        # Query projection (optionally with LoRA)
        if q_lora_rank > 0:
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        
        # KV compression projection
        self.kv_a_proj = nn.Linear(hidden_size, kv_lora_rank + self.rope_dim, bias=False)
        self.kv_b_proj = nn.Linear(kv_lora_rank, self.num_kv_heads * self.head_dim * 2, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = YaRNRotaryEmbedding(
            self.rope_dim,
            max_position_embeddings=max_position_embeddings,
        )
        
        self.kv_norm = RMSNorm(kv_lora_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Query projection
        if self.q_lora_rank > 0:
            q = self.q_b_proj(self.q_a_proj(hidden_states))
        else:
            q = self.q_proj(hidden_states)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # KV compression
        kv_compress = self.kv_a_proj(hidden_states)
        kv_pe, kv_latent = kv_compress.split([self.rope_dim, self.kv_lora_rank], dim=-1)
        kv_latent = self.kv_norm(kv_latent)
        
        # Decompress KV
        kv = self.kv_b_proj(kv_latent)
        kv = kv.view(batch_size, seq_len, self.num_kv_heads, 2, self.head_dim)
        k, v = kv.unbind(dim=3)
        
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        
        cos, sin = self.rotary_emb(q, position_ids)
        
        # Apply rotary to query and key (only rope_dim portion)
        q_pe = q[..., :self.rope_dim]
        k_pe = k[..., :self.rope_dim]
        
        q_pe_rot = q_pe * cos.unsqueeze(1) + self._rotate_half(q_pe) * sin.unsqueeze(1)
        k_pe_rot = k_pe * cos.unsqueeze(1) + self._rotate_half(k_pe) * sin.unsqueeze(1)
        
        q = torch.cat([q_pe_rot, q[..., self.rope_dim:]], dim=-1)
        k = torch.cat([k_pe_rot, k[..., self.rope_dim:]], dim=-1)
        
        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        past_key_value = (k, v) if use_cache else None
        
        # GQA: repeat KV heads
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None if not output_attentions else attn_weights, past_key_value
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class MoEGate(nn.Module):
    """Mixture of Experts gating with Aux-Lossless routing."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        use_shared_expert: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_shared_expert = use_shared_expert
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            expert_weights: [batch_size * seq_len, num_experts_per_tok]
            expert_indices: [batch_size * seq_len, num_experts_per_tok]
            router_logits: [batch_size * seq_len, num_experts]
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        
        router_logits = self.gate(hidden_states_flat)
        
        # Top-k routing
        topk_weights, topk_indices = torch.topk(
            router_logits, 
            self.num_experts_per_tok, 
            dim=-1
        )
        
        # Normalize weights
        expert_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        
        return expert_weights, topk_indices, router_logits


class MoEMLP(nn.Module):
    """Single expert MLP block."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEBlock(nn.Module):
    """Mixture of Experts block with Aux-Lossless routing."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        use_shared_expert: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        
        # Router
        self.gate = MoEGate(hidden_size, num_experts, num_experts_per_tok, use_shared_expert)
        
        # Experts
        self.experts = nn.ModuleList([
            MoEMLP(hidden_size, intermediate_size) 
            for _ in range(num_experts)
        ])
        
        # Shared expert (if enabled)
        self.shared_expert = MoEMLP(hidden_size, intermediate_size) if use_shared_expert else None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing
        expert_weights, expert_indices, router_logits = self.gate(hidden_states)
        
        # Flatten for processing
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Initialize output
        output = torch.zeros_like(hidden_states_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            if expert_mask.any():
                expert_input = hidden_states_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                
                # Get weights for this expert
                weight_mask = (expert_indices[expert_mask] == expert_idx)
                weights = (expert_weights[expert_mask] * weight_mask.float()).sum(dim=-1, keepdim=True)
                
                output[expert_mask] += expert_output * weights
        
        # Add shared expert
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states_flat)
            output = output + shared_output
        
        output = output.view(batch_size, seq_len, hidden_size)
        
        # Aux-lossless: no auxiliary loss
        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        
        return output, aux_loss


class XoronDecoderLayer(nn.Module):
    """Single decoder layer for Xoron model."""
    
    def __init__(
        self,
        config: XoronConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Self-attention
        self.self_attn = MultiHeadLatentAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            kv_lora_rank=getattr(config, 'kv_lora_rank', 512),
            rope_dim=getattr(config, 'qk_rope_head_dim', 64),
            max_position_embeddings=config.max_position_embeddings,
        )
        
        # MoE or regular MLP (alternating based on moe_layer_freq)
        use_moe = config.use_moe and (layer_idx % config.moe_layer_freq == 0)
        if use_moe:
            self.mlp = MoEBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                use_shared_expert=config.use_shared_expert,
            )
        else:
            self.mlp = MoEMLP(config.hidden_size, config.intermediate_size)
        
        self.is_moe = use_moe
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        
        # MLP/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        aux_loss = None
        if self.is_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights, present_key_value, aux_loss


class XoronModel(nn.Module):
    """Xoron base transformer model."""
    
    def __init__(self, config: XoronConfig):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        
        self.layers = nn.ModuleList([
            XoronDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Count MoE layers
        self.num_moe_layers = sum(1 for layer in self.layers if layer.is_moe)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List], Optional[Tuple], Optional[Tuple], Optional[torch.Tensor]]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_len = inputs_embeds.shape[:2]
        
        # Position IDs
        if position_ids is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_length, past_length + seq_len, 
                device=inputs_embeds.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Attention mask
        if attention_mask is not None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=inputs_embeds.device) * float('-inf'),
                diagonal=1
            )
            if attention_mask.dim() == 2:
                # Expand attention mask
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * float('-inf')
            attention_mask = attention_mask + causal_mask
        
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = [] if use_cache else None
        total_aux_loss = torch.tensor(0.0, device=inputs_embeds.device)
        
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            hidden_states, attn_weights, present_key_value, aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            if use_cache:
                next_cache.append(present_key_value)
            
            if output_attentions:
                all_attentions += (attn_weights,)
            
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return hidden_states, next_cache, all_hidden_states, all_attentions, total_aux_loss


class XoronForCausalLM(nn.Module):
    """
    Xoron Model for Causal Language Modeling with multimodal support.
    
    This is the main entry point for using Xoron with kt-kernel. It provides:
    - Full text generation capabilities with MoE/MLA
    - Vision understanding (images and videos)
    - Audio understanding and generation
    - Image and video generation
    
    Usage:
        model = XoronForCausalLM.from_pretrained("your-repo/xoron-model")
        
        # Text generation
        output = model.generate(input_ids, max_new_tokens=100)
        
        # Multimodal inference
        output = model(
            input_ids=tokens,
            pixel_values=images,
            video_frames=videos,
            audio_features=audio,
        )
    """
    
    def __init__(self, config: XoronConfig):
        super().__init__()
        self.config = config
        
        # Core LLM
        self.model = XoronModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        
        # Vision encoder (lazy loaded)
        self._vision_encoder = None
        self._vision_projector = None
        
        # Video encoder (lazy loaded)
        self._video_encoder = None
        
        # Audio encoder/decoder (lazy loaded)
        self._audio_encoder = None
        self._audio_decoder = None
        
        # Image generator (lazy loaded)
        self._image_generator = None
        
        # Video generator (lazy loaded)
        self._video_generator = None
        
        # Modality markers
        self.image_start = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.image_end = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.video_start = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.video_end = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.audio_start = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.audio_end = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        
        # kt-kernel MoE wrapper (will be set during inference setup)
        self._kt_moe_wrapper = None

    @property
    def vision_encoder(self):
        """Lazy load vision encoder."""
        if self._vision_encoder is None and self.config.has_vision_encoder:
            self._init_vision_encoder()
        return self._vision_encoder

    @property
    def audio_encoder(self):
        """Lazy load audio encoder."""
        if self._audio_encoder is None and self.config.has_audio_encoder:
            self._init_audio_encoder()
        return self._audio_encoder

    def _init_vision_encoder(self):
        """Initialize vision encoder components."""
        from .encoders import VisionEncoder, VisionProjector
        
        self._vision_encoder = VisionEncoder(
            model_name=self.config.vision_model_name,
            use_titok=self.config.use_vision_titok,
            num_titok_tokens=self.config.num_vision_titok_tokens,
            freeze=self.config.freeze_vision,
        )
        self._vision_projector = VisionProjector(
            vision_hidden_size=self._vision_encoder.hidden_size,
            llm_hidden_size=self.config.hidden_size,
            num_tokens=self.config.num_vision_tokens,
        )

    def _init_audio_encoder(self):
        """Initialize audio encoder components."""
        from .encoders import AudioEncoder, AudioDecoder
        
        self._audio_encoder = AudioEncoder(
            hidden_size=self.config.hidden_size,
            sample_rate=self.config.audio_sample_rate,
            use_raw_waveform=self.config.use_raw_waveform,
        )
        if self.config.has_audio_decoder:
            self._audio_decoder = AudioDecoder(
                hidden_size=self.config.hidden_size,
                sample_rate=self.config.audio_sample_rate,
            )

    def setup_kt_kernel(
        self,
        weight_path: str,
        method: str = "AMXINT4",
        num_gpu_experts: int = 2,
        cpuinfer_threads: int = 32,
        threadpool_count: int = 2,
    ):
        """
        Setup kt-kernel for high-performance CPU/GPU inference.
        
        Args:
            weight_path: Path to quantized weights
            method: Quantization method (AMXINT4, AMXINT8, FP8, BF16, etc.)
            num_gpu_experts: Number of experts to keep on GPU
            cpuinfer_threads: Number of CPU inference threads
            threadpool_count: Number of NUMA pools
        """
        try:
            from kt_kernel import KTMoEWrapper
            
            # Create wrapper for each MoE layer
            self._kt_moe_wrapper = {}
            
            for layer_idx, layer in enumerate(self.model.layers):
                if layer.is_moe:
                    # Create GPU experts mask
                    gpu_mask = torch.zeros(self.config.num_experts, dtype=torch.bool)
                    gpu_mask[:num_gpu_experts] = True
                    
                    self._kt_moe_wrapper[layer_idx] = KTMoEWrapper(
                        layer_idx=layer_idx,
                        num_experts=self.config.num_experts,
                        num_experts_per_tok=self.config.num_experts_per_tok,
                        hidden_size=self.config.hidden_size,
                        moe_intermediate_size=self.config.intermediate_size,
                        gpu_experts_mask=gpu_mask,
                        cpuinfer_threads=cpuinfer_threads,
                        threadpool_count=threadpool_count,
                        weight_path=weight_path,
                        chunked_prefill_size=512,
                        method=method,
                    )
            
            logger.info(f"kt-kernel initialized with {len(self._kt_moe_wrapper)} MoE layers")
            
        except ImportError:
            logger.warning("kt-kernel not available, using standard PyTorch inference")
            self._kt_moe_wrapper = None

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            pixel_values: [B, C, H, W] tensor
            
        Returns:
            [B, num_tokens, hidden_size] tensor
        """
        if self.vision_encoder is None:
            raise RuntimeError("Vision encoder not initialized")
        
        features = self._vision_encoder(pixel_values)
        projected = self._vision_projector(features)
        
        # Add modality markers
        batch_size = pixel_values.shape[0]
        start_marker = self.image_start.expand(batch_size, -1, -1)
        end_marker = self.image_end.expand(batch_size, -1, -1)
        
        return torch.cat([start_marker, projected, end_marker], dim=1)

    def encode_audio(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to embeddings.
        
        Args:
            audio_waveform: [B, T] or [B, 1, T] raw waveform
            
        Returns:
            [B, num_tokens, hidden_size] tensor
        """
        if self.audio_encoder is None:
            raise RuntimeError("Audio encoder not initialized")
        
        features = self._audio_encoder(audio_waveform)
        
        # Add modality markers
        batch_size = audio_waveform.shape[0]
        start_marker = self.audio_start.expand(batch_size, -1, -1)
        end_marker = self.audio_end.expand(batch_size, -1, -1)
        
        return torch.cat([start_marker, features, end_marker], dim=1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, XoronModelOutput]:
        """
        Forward pass for the Xoron model.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            position_ids: Position IDs [B, L]
            inputs_embeds: Direct embeddings input [B, L, D]
            pixel_values: Image tensors [B, C, H, W]
            video_frames: Video tensors [B, T, C, H, W]
            audio_features: Audio tensors [B, T] or [B, 1, T]
            labels: Target labels for loss computation
            past_key_values: Cached key-values for generation
            use_cache: Whether to return key-values
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return XoronModelOutput
        """
        # Process multimodal inputs
        multimodal_embeds = []
        
        if pixel_values is not None:
            image_embeds = self.encode_image(pixel_values)
            multimodal_embeds.append(image_embeds)
        
        if audio_features is not None:
            audio_embeds = self.encode_audio(audio_features)
            multimodal_embeds.append(audio_embeds)
        
        # Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        
        # Combine multimodal and text embeddings
        if multimodal_embeds:
            # Prepend multimodal tokens before text
            all_embeds = multimodal_embeds + [inputs_embeds]
            inputs_embeds = torch.cat(all_embeds, dim=1)
            
            # Adjust attention mask if provided
            if attention_mask is not None:
                multimodal_len = sum(e.shape[1] for e in multimodal_embeds)
                multimodal_mask = torch.ones(
                    attention_mask.shape[0], 
                    multimodal_len,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([multimodal_mask, attention_mask], dim=1)
        
        # Run through transformer
        hidden_states, past_key_values, all_hidden_states, all_attentions, aux_loss = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            return (loss, logits, past_key_values, all_hidden_states, all_attentions)
        
        return XoronModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            aux_loss=aux_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text tokens autoregressively.
        
        Args:
            input_ids: Starting token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            attention_mask: Attention mask
            pixel_values: Optional image input
            audio_features: Optional audio input
            
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        past_key_values = None
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Handle multimodal inputs on first forward
        first_forward = True
        
        for _ in range(max_new_tokens):
            if first_forward:
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    audio_features=audio_features,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                first_forward = False
            else:
                outputs = self.forward(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1)
            
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        
        return input_ids

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.float16,
        device_map: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "XoronForCausalLM":
        """
        Load a pretrained Xoron model from local path or HuggingFace hub.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            torch_dtype: Data type for model weights
            device_map: Device mapping strategy
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments passed to config
            
        Returns:
            XoronForCausalLM instance
        """
        from pathlib import Path
        
        model_path = Path(model_path)
        
        # Load config
        config = XoronConfig.from_pretrained(str(model_path), **kwargs)
        
        # Create model
        model = cls(config)
        
        # Load weights
        if model_path.exists():
            # Local path
            weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
            if weight_files:
                from safetensors.torch import load_file
                
                state_dict = {}
                for wf in weight_files:
                    if wf.suffix == '.safetensors':
                        state_dict.update(load_file(str(wf)))
                    else:
                        state_dict.update(torch.load(str(wf), map_location='cpu'))
                
                model.load_state_dict(state_dict, strict=False)
        else:
            # HuggingFace hub
            try:
                from huggingface_hub import snapshot_download
                local_path = snapshot_download(repo_id=str(model_path))
                return cls.from_pretrained(local_path, torch_dtype=torch_dtype, **kwargs)
            except Exception as e:
                raise ValueError(f"Could not load model from {model_path}: {e}")
        
        # Convert dtype
        model = model.to(torch_dtype)
        
        # Device placement
        if device_map == "auto":
            try:
                from accelerate import infer_auto_device_map, dispatch_model
                device_map = infer_auto_device_map(model)
                model = dispatch_model(model, device_map)
            except ImportError:
                model = model.cuda() if torch.cuda.is_available() else model
        elif device_map is not None:
            model = model.to(device_map)
        
        return model

    def save_pretrained(self, save_directory: str):
        """
        Save model and config to a directory.
        
        Args:
            save_directory: Directory to save model
        """
        from pathlib import Path
        from safetensors.torch import save_file
        
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(str(save_directory))
        
        # Save weights
        state_dict = self.state_dict()
        save_file(state_dict, str(save_directory / "model.safetensors"))
        
        logger.info(f"Model saved to {save_directory}")


# Register with transformers AutoModel if available
try:
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.register(XoronConfig, XoronForCausalLM)
except ImportError:
    pass
