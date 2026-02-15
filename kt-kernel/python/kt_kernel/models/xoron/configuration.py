# Xoron Model Configuration
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the Xoron Multimodal Model.

This module defines the XoronConfig class which stores all parameters needed
to instantiate a Xoron model, compatible with HuggingFace transformers.
"""

from typing import List, Tuple, Union, Optional
from dataclasses import dataclass, field


@dataclass
class XoronConfig:
    """
    Configuration class for Xoron multimodal model.

    This configuration stores all parameters needed to instantiate the Xoron
    model architecture, including LLM, vision, video, audio, and generation settings.

    SOTA Features:
        - MLA (Multi-Head Latent Attention) for compressed KV cache
        - MoE with shared expert isolation (DeepSeek-style)
        - Ring Attention for distributed 128K+ context
        - YaRN/LongRoPE for superior long-context extrapolation
        - TiTok-style 1D tokenization for vision/video
        - Conformer audio encoder/decoder
        - MoE-DiT with Flow Matching for image generation
        - 3D-RoPE for video generation
    """

    model_type: str = "xoron"
    
    # Model identification
    model_name: str = "Xoron-Dev-MultiMoE"

    # LLM Architecture
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    intermediate_size: int = 2048
    vocab_size: int = 151643
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6

    # Ring Attention
    use_ring_attention: bool = True
    ring_attention_chunk_size: int = 4096

    # Tie word embeddings
    tie_word_embeddings: bool = True

    # MoE Configuration
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_layer_freq: int = 2
    use_shared_expert: bool = True
    moe_capacity_factor: float = 1.25
    use_aux_lossless: bool = True

    # Vision Configuration
    vision_model_name: str = "google/siglip-so400m-patch14-384"
    freeze_vision: bool = False
    num_vision_tokens: int = 64
    projector_type: str = "perceiver"

    # Vision Encoder SOTA Features
    use_vision_dual_stream: bool = True
    use_vision_titok: bool = True
    num_vision_titok_tokens: int = 256
    num_vision_dual_stream_layers: int = 2

    # Video Encoder SOTA Features
    use_video_3d_rope: bool = True
    use_video_temporal_moe: bool = True
    num_video_encoder_layers: int = 4
    num_video_experts: int = 4
    use_video_vidtok: bool = True
    vidtok_latent_channels: int = 4
    vidtok_temporal_compression: int = 4
    vidtok_spatial_compression: int = 8
    vidtok_causal: bool = True

    # VideoTiTokTokenizer Configuration
    use_video_titok: bool = True
    num_video_titok_tokens: int = 64
    num_video_titok_layers: int = 2
    num_video_titok_heads: int = 8
    video_titok_dropout: float = 0.1
    video_max_frames: int = 24

    # Multi-scale training
    use_multi_scale: bool = True
    image_min_size: int = 128
    image_max_size: int = 384
    image_base_size: int = 256
    video_min_size: int = 128
    video_max_size: int = 320

    # Image Generation Configuration
    enable_generation: bool = True
    generation_latent_channels: int = 4
    generation_base_channels: int = 128
    generation_inference_steps: int = 50
    generation_cfg_scale: float = 7.5
    generation_use_flow_matching: bool = True
    generation_num_experts: int = 4
    generation_use_dual_stream: bool = True

    # Video Generation Configuration
    generation_video_cfg_scale: float = 7.5
    generation_video_use_flow_matching: bool = True
    generation_video_num_experts: int = 4
    generation_video_use_3d_rope: bool = True
    generation_video_use_temporal_moe: bool = True

    # Audio Configuration
    audio_sample_rate: int = 16000
    audio_n_mels: int = 80
    audio_max_length: int = 625
    audio_max_waveform_samples: int = 160000
    audio_num_speakers: int = 256
    use_raw_waveform: bool = True
    audio_kv_lora_rank: int = 256
    audio_speaker_embed_dim: int = 256
    use_mas: bool = True
    use_in_context_audio_prompting: bool = True

    # Tokenizer Configuration
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B"

    # LoRA Configuration
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    )
    train_lora_only: bool = False
    use_rslora: bool = True
    use_dora: bool = False

    # Cross-Attention Configuration
    use_cross_attention: bool = True
    cross_attention_layers: int = 4
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1

    # Flash Attention Configuration
    use_flash_attention: bool = True

    # Architecture flags
    has_audio_encoder: bool = True
    has_audio_decoder: bool = True
    has_waveform_decoder: bool = True
    has_vision_encoder: bool = True
    has_video_encoder: bool = True
    has_generator: bool = True
    has_video_generator: bool = True
    has_cross_attention: bool = True
    lora_applied: bool = False
    architecture_version: int = 2

    # kt-kernel specific settings
    kt_method: str = "AMXINT4"
    kt_num_gpu_experts: int = 2
    kt_cpuinfer_threads: int = 32
    kt_threadpool_count: int = 2

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) cannot exceed "
                f"num_experts ({self.num_experts})"
            )
        
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "XoronConfig":
        """
        Load configuration from a pretrained model directory or HuggingFace hub.
        
        Args:
            model_path: Path to local model or HuggingFace model ID
            **kwargs: Override any config parameters
            
        Returns:
            XoronConfig instance
        """
        import json
        import os
        from pathlib import Path
        
        model_path = Path(model_path)
        
        # Check local path first
        if model_path.exists():
            config_file = model_path / "config.json"
        else:
            # Try HuggingFace hub
            try:
                from huggingface_hub import hf_hub_download
                config_file = Path(hf_hub_download(
                    repo_id=str(model_path),
                    filename="config.json",
                ))
            except Exception as e:
                raise ValueError(f"Could not load config from {model_path}: {e}")
        
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        
        # Filter out unknown keys
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_keys}
        
        # Apply overrides
        filtered_dict.update(kwargs)
        
        return cls(**filtered_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save_pretrained(self, save_directory: str):
        """Save configuration to a directory."""
        import json
        import os
        from pathlib import Path
        
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        config_file = save_directory / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_moe_config(self) -> dict:
        """Get MoE-specific configuration for kt-kernel."""
        return {
            "use_moe": self.use_moe,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_layer_freq": self.moe_layer_freq,
            "intermediate_size": self.intermediate_size,
            "hidden_size": self.hidden_size,
            "use_shared_expert": self.use_shared_expert,
            "use_aux_lossless": self.use_aux_lossless,
        }

    def get_kt_kernel_config(self) -> dict:
        """Get kt-kernel specific configuration for inference."""
        return {
            "method": self.kt_method,
            "num_gpu_experts": self.kt_num_gpu_experts,
            "cpuinfer_threads": self.kt_cpuinfer_threads,
            "threadpool_count": self.kt_threadpool_count,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
        }


# Register with transformers AutoConfig if available
try:
    from transformers import AutoConfig
    AutoConfig.register("xoron", XoronConfig)
except ImportError:
    pass
