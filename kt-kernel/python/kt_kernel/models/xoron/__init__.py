# Xoron Model Support for kt-kernel
# SPDX-License-Identifier: Apache-2.0

"""
Xoron Multimodal Model Support for kt-kernel.

This package provides full integration for the Xoron multimodal model,
including:
- Text understanding and generation (MoE LLM with MLA)
- Image understanding (SigLIP encoder + TiTok)
- Image generation (MoE-DiT with Flow Matching)
- Video understanding (VideoTiTok + Temporal MoE)
- Video generation (3D-RoPE + Temporal MoE)
- Audio understanding (Raw Waveform + Conformer)
- Audio generation (TTS with zero-shot speaker cloning)

Usage:
    from kt_kernel.models.xoron import XoronForCausalLM, XoronMultimodalProcessor
    
    # Load from HuggingFace
    model = XoronForCausalLM.from_pretrained("your-repo/xoron-model")
    processor = XoronMultimodalProcessor.from_pretrained("your-repo/xoron-model")
"""

from .configuration import XoronConfig
from .modeling import XoronForCausalLM
from .processing import XoronMultimodalProcessor
from .loader import XoronModelLoader

__all__ = [
    "XoronConfig",
    "XoronForCausalLM", 
    "XoronMultimodalProcessor",
    "XoronModelLoader",
]
