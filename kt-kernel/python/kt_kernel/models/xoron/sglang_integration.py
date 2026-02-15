# Xoron SGLang Integration
# SPDX-License-Identifier: Apache-2.0

"""
SGLang integration for Xoron multimodal model.

This module provides the necessary hooks and configurations to deploy
Xoron as an API server using SGLang with kt-kernel acceleration.

Usage:
    # Via CLI
    kt run your-repo/xoron-model --kt-method AMXINT4 --kt-num-gpu-experts 2
    
    # Via Python
    from kt_kernel.models.xoron import start_xoron_server
    start_xoron_server("your-repo/xoron-model", port=8000)
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def get_xoron_server_args(
    model_path: str,
    kt_weight_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    kt_method: str = "AMXINT4",
    kt_num_gpu_experts: int = 2,
    kt_cpuinfer_threads: int = 32,
    kt_threadpool_count: int = 2,
    tensor_parallel_size: int = 1,
    max_total_tokens: int = 131072,
    max_running_requests: int = 8,
    chunked_prefill_size: int = 512,
    mem_fraction_static: float = 0.85,
    **kwargs,
) -> List[str]:
    """
    Build SGLang server arguments for Xoron model.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        kt_weight_path: Path to quantized weights (optional)
        host: Server host address
        port: Server port
        kt_method: kt-kernel quantization method
        kt_num_gpu_experts: Number of GPU experts
        kt_cpuinfer_threads: CPU inference threads
        kt_threadpool_count: NUMA pool count
        tensor_parallel_size: Tensor parallel size
        max_total_tokens: Maximum total tokens
        max_running_requests: Maximum running requests
        chunked_prefill_size: Chunked prefill size
        mem_fraction_static: Static memory fraction
        **kwargs: Additional SGLang arguments
        
    Returns:
        List of command-line arguments for SGLang
    """
    args = [
        sys.executable,
        "-m", "sglang.launch_server",
        "--host", host,
        "--port", str(port),
        "--model", str(model_path),
        "--trust-remote-code",
    ]
    
    # kt-kernel options
    weight_path = kt_weight_path or model_path
    args.extend([
        "--kt-weight-path", str(weight_path),
        "--kt-method", kt_method,
        "--kt-num-gpu-experts", str(kt_num_gpu_experts),
        "--kt-cpuinfer", str(kt_cpuinfer_threads),
        "--kt-threadpool-count", str(kt_threadpool_count),
        "--kt-gpu-prefill-token-threshold", "256",
        "--kt-enable-dynamic-expert-update",
    ])
    
    # SGLang options
    args.extend([
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-total-tokens", str(max_total_tokens),
        "--max-running-requests", str(max_running_requests),
        "--chunked-prefill-size", str(chunked_prefill_size),
        "--mem-fraction-static", str(mem_fraction_static),
        "--attention-backend", "flashinfer",
        "--enable-mixed-chunk",
        "--enable-p2p-check",
    ])
    
    # Additional arguments
    for key, value in kwargs.items():
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(arg_name)
        else:
            args.extend([arg_name, str(value)])
    
    return args


def start_xoron_server(
    model_path: str,
    **kwargs,
) -> None:
    """
    Start Xoron model server using SGLang.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        **kwargs: Arguments passed to get_xoron_server_args()
    """
    import subprocess
    
    args = get_xoron_server_args(model_path, **kwargs)
    
    logger.info(f"Starting Xoron server with command: {' '.join(args)}")
    
    # Set environment variables
    env = os.environ.copy()
    env["XORON_MODEL_TYPE"] = "multimodal"
    
    try:
        process = subprocess.run(args, env=env)
        sys.exit(process.returncode)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


class XoronSGLangConfig:
    """
    Configuration class for Xoron SGLang deployment.
    
    This class provides model-specific configurations that SGLang needs
    to properly serve the Xoron model.
    """
    
    # Model architecture info
    model_type = "xoron"
    architectures = ["XoronForCausalLM"]
    
    # Multimodal capabilities
    supports_vision = True
    supports_video = True
    supports_audio = True
    supports_image_generation = True
    supports_video_generation = True
    supports_audio_generation = True
    
    # Special tokens
    image_token = "<|image|>"
    video_token = "<|video|>"
    audio_token = "<|audio|>"
    
    # Default generation parameters
    default_temperature = 0.7
    default_top_p = 0.9
    default_top_k = 50
    default_max_tokens = 2048
    
    # MoE settings
    is_moe = True
    moe_aux_lossless = True
    
    @classmethod
    def get_model_kwargs(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get model initialization kwargs from config."""
        return {
            "trust_remote_code": True,
            "torch_dtype": "float16",
        }
    
    @classmethod
    def get_sampling_params(cls, **kwargs) -> Dict[str, Any]:
        """Get default sampling parameters."""
        return {
            "temperature": kwargs.get("temperature", cls.default_temperature),
            "top_p": kwargs.get("top_p", cls.default_top_p),
            "top_k": kwargs.get("top_k", cls.default_top_k),
            "max_tokens": kwargs.get("max_tokens", cls.default_max_tokens),
        }


class XoronMultimodalEndpoint:
    """
    Multimodal endpoint handler for Xoron model.
    
    Handles requests that include images, videos, or audio along with text.
    """
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    async def process_request(
        self,
        text: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process a multimodal request.
        
        Args:
            text: Input text prompt
            images: List of image paths or URLs
            videos: List of video paths or URLs
            audio: List of audio paths or URLs
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with generated outputs
        """
        import torch
        
        # Process inputs
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            audio=audio,
            return_tensors="pt",
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            if "generate_image" in kwargs and kwargs["generate_image"]:
                # Image generation
                output = self.model.generate_image(**inputs)
                return {"type": "image", "content": output}
            
            elif "generate_video" in kwargs and kwargs["generate_video"]:
                # Video generation
                output = self.model.generate_video(**inputs)
                return {"type": "video", "content": output}
            
            elif "generate_speech" in kwargs and kwargs["generate_speech"]:
                # Speech generation
                output = self.model.generate_speech(**inputs)
                return {"type": "audio", "content": output}
            
            else:
                # Text generation
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", 2048),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    do_sample=kwargs.get("do_sample", True),
                )
                
                text_output = self.processor.decode(output_ids[0])
                return {"type": "text", "content": text_output}


def register_xoron_with_sglang():
    """
    Register Xoron model with SGLang's model registry.
    
    This allows SGLang to automatically recognize and handle Xoron models.
    """
    try:
        # Try to import SGLang's model registry
        from sglang.srt.models.registry import ModelRegistry
        
        # Register Xoron model
        ModelRegistry.register(
            "xoron",
            "kt_kernel.models.xoron.XoronForCausalLM",
            architectures=["XoronForCausalLM"],
        )
        
        logger.info("Xoron model registered with SGLang")
        
    except ImportError:
        logger.warning("SGLang not installed, skipping model registration")
    except Exception as e:
        logger.warning(f"Could not register Xoron with SGLang: {e}")


# API endpoint definitions for OpenAI-compatible API
XORON_API_ENDPOINTS = {
    # Chat completions (standard OpenAI)
    "/v1/chat/completions": {
        "method": "POST",
        "description": "Create chat completion with optional multimodal inputs",
        "supports": ["text", "vision", "audio"],
    },
    
    # Image generation
    "/v1/images/generations": {
        "method": "POST",
        "description": "Generate images from text prompts",
        "supports": ["text-to-image"],
    },
    
    # Audio
    "/v1/audio/transcriptions": {
        "method": "POST",
        "description": "Transcribe audio to text",
        "supports": ["audio-to-text"],
    },
    "/v1/audio/speech": {
        "method": "POST",
        "description": "Generate speech from text",
        "supports": ["text-to-speech"],
    },
    
    # Video (Xoron-specific)
    "/v1/video/generations": {
        "method": "POST",
        "description": "Generate videos from text prompts",
        "supports": ["text-to-video"],
    },
    "/v1/video/understanding": {
        "method": "POST",
        "description": "Understand video content",
        "supports": ["video-to-text"],
    },
}


def get_openai_compatible_config() -> Dict[str, Any]:
    """
    Get OpenAI-compatible API configuration for Xoron.
    
    Returns:
        Configuration dict for OpenAI-compatible endpoints
    """
    return {
        "model": "xoron",
        "endpoints": XORON_API_ENDPOINTS,
        "capabilities": {
            "chat": True,
            "completion": True,
            "vision": True,
            "audio": True,
            "video": True,
            "image_generation": True,
            "audio_generation": True,
            "video_generation": True,
        },
        "context_length": 131072,
        "max_output_tokens": 8192,
    }


# Auto-register on import
try:
    register_xoron_with_sglang()
except Exception:
    pass
