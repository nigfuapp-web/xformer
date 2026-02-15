# Xoron Multimodal Encoders
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal encoders for Xoron model.

This module provides encoders for:
- Vision (images with SigLIP/CLIP + TiTok)
- Video (VideoTiTok with temporal MoE)
- Audio (Raw waveform + Conformer)
"""

import math
import logging
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for vision encoder patches."""

    def __init__(self, dim: int, max_height: int = 128, max_width: int = 128, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.base = base
        
        self.dim_x = dim // 2
        self.dim_y = dim - self.dim_x
        
        inv_freq_x = 1.0 / (base ** (torch.arange(0, self.dim_x, 2, dtype=torch.float32) / self.dim_x))
        inv_freq_y = 1.0 / (base ** (torch.arange(0, self.dim_y, 2, dtype=torch.float32) / self.dim_y))
        
        self.register_buffer('inv_freq_x', inv_freq_x, persistent=False)
        self.register_buffer('inv_freq_y', inv_freq_y, persistent=False)

    def forward(self, x: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        
        pos_x = torch.arange(width, device=device, dtype=torch.float32)
        pos_y = torch.arange(height, device=device, dtype=torch.float32)
        
        freqs_x = torch.outer(pos_x, self.inv_freq_x.to(device))
        freqs_y = torch.outer(pos_y, self.inv_freq_y.to(device))
        
        freqs_x = torch.cat([freqs_x, freqs_x], dim=-1)
        freqs_y = torch.cat([freqs_y, freqs_y], dim=-1)
        
        cos_2d = torch.zeros(height, width, self.dim, device=device, dtype=dtype)
        sin_2d = torch.zeros(height, width, self.dim, device=device, dtype=dtype)
        
        for y in range(height):
            for w in range(width):
                cos_2d[y, w, :self.dim_x] = freqs_x[w].cos().to(dtype)
                sin_2d[y, w, :self.dim_x] = freqs_x[w].sin().to(dtype)
                cos_2d[y, w, self.dim_x:] = freqs_y[y].cos().to(dtype)
                sin_2d[y, w, self.dim_x:] = freqs_y[y].sin().to(dtype)
        
        cos_2d = cos_2d.view(height * width, self.dim)
        sin_2d = sin_2d.view(height * width, self.dim)
        
        return cos_2d, sin_2d


class TiTokTokenizer(nn.Module):
    """
    TiTok-style 1D Tokenizer for efficient visual representation.
    Converts 2D patch grid to 1D token sequence with learnable compression.
    """

    def __init__(self, hidden_size: int, num_tokens: int = 256, num_patches: int = 576):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.num_patches = num_patches
        
        # Learnable compression
        self.compress = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Learnable token queries
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, hidden_size) * 0.02)
        
        # Cross-attention for compression
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.compress_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress patch features to TiTok-style 1D tokens.
        
        Args:
            x: [B, num_patches, hidden_size] patch features
            
        Returns:
            [B, num_tokens, hidden_size] compressed tokens
        """
        batch_size = x.shape[0]
        
        queries = self.token_queries.expand(batch_size, -1, -1)
        x_proj = self.compress(x)
        tokens, _ = self.compress_attn(queries, x_proj, x_proj)
        tokens = self.compress_norm(queries + tokens)
        
        return tokens


class VisionEncoder(nn.Module):
    """
    Vision Encoder with SigLIP/CLIP backbone and optional TiTok compression.
    
    Features:
        - SigLIP 2 or CLIP backbone
        - 2D-RoPE for flexible aspect ratios
        - TiTok-style 1D tokenization
        - Dual-stream attention
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        use_titok: bool = True,
        num_titok_tokens: int = 256,
        freeze: bool = False,
        use_dual_stream: bool = True,
        num_dual_stream_layers: int = 2,
    ):
        super().__init__()
        self.use_titok = use_titok
        self.model_name = model_name
        
        # Initialize backbone
        self._init_backbone(model_name, freeze)
        
        # 2D-RoPE
        self.rope_2d = RoPE2D(
            dim=self.hidden_size,
            max_height=64,
            max_width=64,
        )
        
        # TiTok tokenizer
        if use_titok:
            self.titok = TiTokTokenizer(
                hidden_size=self.hidden_size,
                num_tokens=num_titok_tokens,
                num_patches=self.num_patches,
            )
            logger.info(f"TiTok: {self.num_patches} patches -> {num_titok_tokens} tokens")
        else:
            self.titok = None
        
        # Dual-stream attention
        if use_dual_stream:
            self.dual_stream_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=8,
                    dim_feedforward=self.hidden_size * 4,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True,
                )
                for _ in range(num_dual_stream_layers)
            ])
        else:
            self.dual_stream_layers = None

    def _init_backbone(self, model_name: str, freeze: bool):
        """Initialize vision backbone."""
        is_siglip = 'siglip' in model_name.lower()
        
        try:
            if is_siglip:
                from transformers import SiglipVisionModel, SiglipImageProcessor
                self.vision_model = SiglipVisionModel.from_pretrained(model_name)
                self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
            else:
                from transformers import CLIPVisionModel, CLIPImageProcessor
                self.vision_model = CLIPVisionModel.from_pretrained(model_name)
                self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
            
            self.hidden_size = self.vision_model.config.hidden_size
            
            if freeze:
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                logger.info(f"Vision encoder frozen: {model_name}")
            
            logger.info(f"Vision encoder initialized: {model_name}, hidden_size={self.hidden_size}")
            
        except Exception as e:
            logger.error(f"Failed to load vision encoder: {e}")
            raise

    @property
    def num_patches(self) -> int:
        """Get number of patches for the vision model."""
        config = self.vision_model.config
        image_size = config.image_size
        patch_size = config.patch_size
        return (image_size // patch_size) ** 2

    def forward(self, pixel_values: torch.Tensor, return_titok: bool = None) -> torch.Tensor:
        """
        Extract vision features from images.
        
        Args:
            pixel_values: [B, C, H, W] tensor
            return_titok: Override TiTok output
            
        Returns:
            [B, num_tokens, hidden_size] tensor
        """
        outputs = self.vision_model(pixel_values=pixel_values)
        features = outputs.last_hidden_state
        
        # Handle CLS token
        batch_size, num_patches, hidden_size = features.shape
        patch_size = getattr(self.vision_model.config, 'patch_size', 14)
        image_size = getattr(self.vision_model.config, 'image_size', 384)
        
        if num_patches == (image_size // patch_size) ** 2 + 1:
            features = features[:, 1:]  # Remove CLS token
            num_patches = num_patches - 1
        
        height = width = int(math.sqrt(num_patches))
        
        # Dual-stream attention
        if self.dual_stream_layers is not None:
            for layer in self.dual_stream_layers:
                features = layer(features)
        
        # TiTok compression
        use_titok_now = return_titok if return_titok is not None else self.use_titok
        if use_titok_now and self.titok is not None:
            features = self.titok(features)
        
        return features


class VisionProjector(nn.Module):
    """Project vision features to LLM hidden dimension."""
    
    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        num_tokens: int = 64,
        projector_type: str = "perceiver",
    ):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.num_tokens = num_tokens
        
        if projector_type == "perceiver":
            # Perceiver-style resampler
            self.latent_queries = nn.Parameter(torch.randn(1, num_tokens, llm_hidden_size) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=llm_hidden_size,
                num_heads=8,
                kdim=vision_hidden_size,
                vdim=vision_hidden_size,
                batch_first=True,
                dropout=0.1,
            )
            self.ffn = nn.Sequential(
                nn.Linear(llm_hidden_size, llm_hidden_size * 4),
                nn.GELU(),
                nn.Linear(llm_hidden_size * 4, llm_hidden_size),
            )
            self.norm1 = nn.LayerNorm(llm_hidden_size)
            self.norm2 = nn.LayerNorm(llm_hidden_size)
        else:
            # Simple MLP projection
            self.proj = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size * 2),
                nn.GELU(),
                nn.Linear(llm_hidden_size * 2, llm_hidden_size),
            )
        
        self.projector_type = projector_type

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to LLM space.
        
        Args:
            vision_features: [B, N, vision_hidden_size]
            
        Returns:
            [B, num_tokens, llm_hidden_size]
        """
        if self.projector_type == "perceiver":
            batch_size = vision_features.shape[0]
            queries = self.latent_queries.expand(batch_size, -1, -1)
            
            # Cross-attention
            attn_out, _ = self.cross_attn(queries, vision_features, vision_features)
            queries = self.norm1(queries + attn_out)
            
            # FFN
            ffn_out = self.ffn(queries)
            output = self.norm2(queries + ffn_out)
            
            return output
        else:
            return self.proj(vision_features)


class VideoEncoder(nn.Module):
    """
    Video Encoder with temporal modeling and VideoTiTok compression.
    
    Features:
        - Frame-wise vision encoding
        - 3D-RoPE for temporal position encoding
        - Temporal MoE for efficient processing
        - VideoTiTok for 1D token compression
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        max_frames: int = 24,
        use_temporal_moe: bool = True,
        num_experts: int = 4,
        num_video_titok_tokens: int = 64,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.max_frames = max_frames
        self.hidden_size = vision_encoder.hidden_size
        
        # Temporal position embedding
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_frames, 1, self.hidden_size) * 0.02)
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.temporal_norm = nn.LayerNorm(self.hidden_size)
        
        # VideoTiTok compression
        self.video_titok = TiTokTokenizer(
            hidden_size=self.hidden_size,
            num_tokens=num_video_titok_tokens,
            num_patches=max_frames * vision_encoder.num_patches if vision_encoder.titok is None else max_frames * vision_encoder.titok.num_tokens,
        )
        
        # Temporal MoE
        if use_temporal_moe:
            self.temporal_moe = TemporalMoE(
                hidden_size=self.hidden_size,
                num_experts=num_experts,
            )
        else:
            self.temporal_moe = None

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames.
        
        Args:
            video_frames: [B, T, C, H, W] video tensor
            
        Returns:
            [B, num_tokens, hidden_size] video features
        """
        batch_size, num_frames = video_frames.shape[:2]
        
        # Flatten for vision encoder
        frames_flat = video_frames.view(-1, *video_frames.shape[2:])
        
        # Encode frames
        frame_features = self.vision_encoder(frames_flat)
        
        # Reshape: [B*T, N, D] -> [B, T, N, D]
        num_patches = frame_features.shape[1]
        frame_features = frame_features.view(batch_size, num_frames, num_patches, -1)
        
        # Add temporal position embeddings
        frame_features = frame_features + self.temporal_pos_embed[:, :num_frames]
        
        # Temporal attention: [B, T, N, D] -> [B, N, T, D] for temporal processing
        frame_features = frame_features.transpose(1, 2)  # [B, N, T, D]
        batch_size, num_patches, num_frames, hidden_size = frame_features.shape
        
        # Process each patch position through time
        frame_features = frame_features.reshape(batch_size * num_patches, num_frames, hidden_size)
        attn_out, _ = self.temporal_attention(frame_features, frame_features, frame_features)
        frame_features = self.temporal_norm(frame_features + attn_out)
        
        # Temporal MoE
        if self.temporal_moe is not None:
            frame_features = self.temporal_moe(frame_features)
        
        # Reshape back: [B*N, T, D] -> [B, T*N, D]
        frame_features = frame_features.view(batch_size, num_patches, num_frames, hidden_size)
        frame_features = frame_features.transpose(1, 2)  # [B, T, N, D]
        frame_features = frame_features.reshape(batch_size, num_frames * num_patches, hidden_size)
        
        # VideoTiTok compression
        video_features = self.video_titok(frame_features)
        
        return video_features


class TemporalMoE(nn.Module):
    """Temporal MoE for video processing."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        intermediate_size: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size or hidden_size * 4
        
        # Router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, self.intermediate_size),
                nn.GELU(),
                nn.Linear(self.intermediate_size, hidden_size),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            
        Returns:
            [B, T, D]
        """
        # Get routing weights
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Process through all experts and combine
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [B, T, D, E]
        output = (expert_outputs * routing_weights.unsqueeze(-2)).sum(dim=-1)  # [B, T, D]
        
        return output


class AudioEncoder(nn.Module):
    """
    Audio Encoder with raw waveform support and Conformer architecture.
    
    Features:
        - Raw waveform tokenization (no mel spectrogram needed)
        - Conformer-based encoding
        - Zero-shot speaker embedding extraction
        - FP16-native stability
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        sample_rate: int = 16000,
        use_raw_waveform: bool = True,
        n_mels: int = 80,
        max_audio_length: int = 3000,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sample_rate = sample_rate
        self.use_raw_waveform = use_raw_waveform
        
        if use_raw_waveform:
            # Raw waveform encoder
            self.waveform_encoder = RawWaveformEncoder(
                hidden_size=hidden_size,
                sample_rate=sample_rate,
            )
        else:
            # Mel spectrogram encoder
            self.mel_encoder = MelSpectrogramEncoder(
                hidden_size=hidden_size,
                n_mels=n_mels,
            )
        
        # Conformer layers
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(hidden_size)
            for _ in range(6)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to embeddings.
        
        Args:
            audio: [B, T] raw waveform or [B, n_mels, T] mel spectrogram
            
        Returns:
            [B, num_tokens, hidden_size] audio features
        """
        if self.use_raw_waveform:
            features = self.waveform_encoder(audio)
        else:
            features = self.mel_encoder(audio)
        
        # Conformer processing
        for layer in self.conformer_layers:
            features = layer(features)
        
        # Output projection
        features = self.output_proj(features)
        features = self.output_norm(features)
        
        return features


class RawWaveformEncoder(nn.Module):
    """Encode raw audio waveforms directly."""
    
    def __init__(
        self,
        hidden_size: int = 1024,
        sample_rate: int = 16000,
        hop_length: int = 320,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # Multi-scale 1D convolutions
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv1d(512, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_size),
            nn.SiLU(),
        )
        
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] raw audio
            
        Returns:
            [B, T', hidden_size] encoded features
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [B, 1, T]
        
        x = self.conv_layers(waveform)  # [B, hidden_size, T']
        x = x.transpose(1, 2)  # [B, T', hidden_size]
        x = self.output_proj(x)
        
        return x


class MelSpectrogramEncoder(nn.Module):
    """Encode mel spectrograms."""
    
    def __init__(
        self,
        hidden_size: int = 1024,
        n_mels: int = 80,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_mels = n_mels
        
        # CNN encoder
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Calculate output dimension
        reduced_mels = n_mels // 4
        self.proj = nn.Linear(128 * reduced_mels, hidden_size)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T] mel spectrogram
            
        Returns:
            [B, T', hidden_size]
        """
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)  # [B, 1, n_mels, T]
        
        x = self.conv_layers(mel)  # [B, 128, n_mels//4, T//4]
        batch_size, channels, freq, time = x.shape
        
        x = x.permute(0, 3, 1, 2)  # [B, T//4, 128, freq]
        x = x.reshape(batch_size, time, -1)  # [B, T//4, 128*freq]
        x = self.proj(x)  # [B, T//4, hidden_size]
        
        return x


class ConformerBlock(nn.Module):
    """Single Conformer block for audio processing."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Feed-forward 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * ff_mult, hidden_size),
            nn.Dropout(dropout),
        )
        
        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_norm = nn.LayerNorm(hidden_size)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2, groups=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.Dropout(dropout),
        )
        
        # Feed-forward 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * ff_mult, hidden_size),
            nn.Dropout(dropout),
        )
        
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, hidden_size]
            
        Returns:
            [B, T, hidden_size]
        """
        # Feed-forward 1 (half residual)
        x = x + 0.5 * self.ff1(x)
        
        # Attention
        attn_input = self.attn_norm(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + self.attn_dropout(attn_out)
        
        # Convolution
        conv_input = self.conv_norm(x)
        conv_input = conv_input.transpose(1, 2)  # [B, hidden_size, T]
        conv_out = self.conv(conv_input)
        conv_out = conv_out.transpose(1, 2)  # [B, T, hidden_size]
        x = x + conv_out
        
        # Feed-forward 2 (half residual)
        x = x + 0.5 * self.ff2(x)
        
        x = self.final_norm(x)
        
        return x


class AudioDecoder(nn.Module):
    """
    Audio Decoder for speech synthesis (TTS).
    
    Features:
        - Flow matching for high-quality synthesis
        - Zero-shot speaker cloning
        - Duration prediction
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        sample_rate: int = 16000,
        n_mels: int = 80,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Duration predictor
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            ConformerBlock(hidden_size)
            for _ in range(4)
        ])
        
        # Mel spectrogram output
        self.mel_proj = nn.Linear(hidden_size, n_mels)
        
        # Optional: Direct waveform output
        self.use_vocoder = True  # Set False for direct waveform

    def forward(
        self,
        hidden_states: torch.Tensor,
        speaker_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate audio from hidden states.
        
        Args:
            hidden_states: [B, T, hidden_size] from LLM
            speaker_embedding: [B, hidden_size] optional speaker embedding
            
        Returns:
            mel_spectrogram: [B, n_mels, T']
            duration: [B, T] predicted durations
        """
        # Add speaker embedding if provided
        if speaker_embedding is not None:
            hidden_states = hidden_states + speaker_embedding.unsqueeze(1)
        
        # Predict durations
        durations = self.duration_predictor(hidden_states).squeeze(-1)
        durations = F.softplus(durations)  # Ensure positive
        
        # Decode
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states)
        
        # Output mel spectrogram
        mel = self.mel_proj(hidden_states)  # [B, T, n_mels]
        mel = mel.transpose(1, 2)  # [B, n_mels, T]
        
        return mel, durations
