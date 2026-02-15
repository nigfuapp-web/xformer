# Xoron Model Loader for kt-kernel
# SPDX-License-Identifier: Apache-2.0

"""
Model loader for Xoron with kt-kernel integration.

This module provides utilities for loading Xoron models from HuggingFace
and setting up kt-kernel for high-performance inference.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch

from .configuration import XoronConfig
from .modeling import XoronForCausalLM
from .processing import XoronMultimodalProcessor

logger = logging.getLogger(__name__)


class XoronModelLoader:
    """
    Utility class for loading Xoron models with kt-kernel support.
    
    This class handles:
    - Loading models from HuggingFace or local paths
    - Setting up kt-kernel for CPU/GPU inference
    - Configuring multimodal components
    
    Usage:
        loader = XoronModelLoader()
        
        # Load from HuggingFace
        model, processor = loader.load("your-repo/xoron-model")
        
        # With kt-kernel quantization
        model, processor = loader.load(
            "your-repo/xoron-model",
            kt_weight_path="/path/to/quantized/weights",
            kt_method="AMXINT4",
            kt_num_gpu_experts=2,
        )
    """
    
    def __init__(self):
        self._cached_models: Dict[str, XoronForCausalLM] = {}
        self._cached_processors: Dict[str, XoronMultimodalProcessor] = {}

    def load(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.float16,
        device_map: Optional[str] = "auto",
        trust_remote_code: bool = True,
        kt_weight_path: Optional[str] = None,
        kt_method: str = "AMXINT4",
        kt_num_gpu_experts: int = 2,
        kt_cpuinfer_threads: int = 32,
        kt_threadpool_count: int = 2,
        use_cache: bool = True,
        **kwargs,
    ) -> tuple[XoronForCausalLM, XoronMultimodalProcessor]:
        """
        Load Xoron model and processor.
        
        Args:
            model_path: HuggingFace model ID or local path
            torch_dtype: Data type for model weights
            device_map: Device mapping strategy ("auto", "cuda", "cpu", etc.)
            trust_remote_code: Whether to trust remote code
            kt_weight_path: Path to quantized weights for kt-kernel
            kt_method: Quantization method (AMXINT4, AMXINT8, FP8, BF16)
            kt_num_gpu_experts: Number of experts to keep on GPU
            kt_cpuinfer_threads: Number of CPU inference threads
            kt_threadpool_count: Number of NUMA pools
            use_cache: Whether to cache loaded models
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (model, processor)
        """
        cache_key = f"{model_path}_{torch_dtype}_{device_map}"
        
        # Check cache
        if use_cache and cache_key in self._cached_models:
            logger.info(f"Using cached model: {model_path}")
            return self._cached_models[cache_key], self._cached_processors.get(cache_key)
        
        # Load model
        logger.info(f"Loading Xoron model from {model_path}")
        
        model = XoronForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        
        # Load processor
        processor = XoronMultimodalProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        
        # Setup kt-kernel if weights provided
        if kt_weight_path is not None:
            logger.info(f"Setting up kt-kernel with method={kt_method}")
            model.setup_kt_kernel(
                weight_path=kt_weight_path,
                method=kt_method,
                num_gpu_experts=kt_num_gpu_experts,
                cpuinfer_threads=kt_cpuinfer_threads,
                threadpool_count=kt_threadpool_count,
            )
        
        # Cache
        if use_cache:
            self._cached_models[cache_key] = model
            self._cached_processors[cache_key] = processor
        
        return model, processor

    def load_for_sglang(
        self,
        model_path: str,
        kt_weight_path: Optional[str] = None,
        kt_method: str = "AMXINT4",
        kt_num_gpu_experts: int = 2,
        kt_cpuinfer_threads: int = 32,
        kt_threadpool_count: int = 2,
        tensor_parallel_size: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare model configuration for SGLang server.
        
        Args:
            model_path: HuggingFace model ID or local path
            kt_weight_path: Path to quantized weights
            kt_method: Quantization method
            kt_num_gpu_experts: Number of GPU experts
            kt_cpuinfer_threads: CPU inference threads
            kt_threadpool_count: NUMA pool count
            tensor_parallel_size: Tensor parallel size
            **kwargs: Additional arguments
            
        Returns:
            Configuration dictionary for SGLang
        """
        config = XoronConfig.from_pretrained(model_path)
        
        sglang_config = {
            "model_path": str(model_path),
            "trust_remote_code": True,
            "tensor_parallel_size": tensor_parallel_size,
            # kt-kernel specific
            "kt_weight_path": kt_weight_path or model_path,
            "kt_method": kt_method,
            "kt_num_gpu_experts": kt_num_gpu_experts,
            "kt_cpuinfer_threads": kt_cpuinfer_threads,
            "kt_threadpool_count": kt_threadpool_count,
            # Model architecture info
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "vocab_size": config.vocab_size,
            # Multimodal capabilities
            "has_vision": config.has_vision_encoder,
            "has_video": config.has_video_encoder,
            "has_audio": config.has_audio_encoder,
            "has_image_generation": config.has_generator,
            "has_video_generation": config.has_video_generator,
            "has_audio_generation": config.has_audio_decoder,
        }
        
        return sglang_config

    def download_from_huggingface(
        self,
        repo_id: str,
        local_dir: Optional[str] = None,
        revision: str = "main",
        use_hf_mirror: bool = False,
    ) -> str:
        """
        Download model from HuggingFace hub.
        
        Args:
            repo_id: HuggingFace repository ID
            local_dir: Local directory to save model
            revision: Git revision to download
            use_hf_mirror: Use HuggingFace mirror
            
        Returns:
            Local path to downloaded model
        """
        from huggingface_hub import snapshot_download
        
        if use_hf_mirror:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        if local_dir is None:
            local_dir = Path.home() / ".cache" / "xoron" / repo_id.replace("/", "_")
        
        logger.info(f"Downloading {repo_id} to {local_dir}")
        
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            revision=revision,
        )
        
        return local_path

    def convert_to_kt_format(
        self,
        model_path: str,
        output_path: str,
        method: str = "AMXINT4",
        numa_nodes: int = 2,
    ) -> str:
        """
        Convert model weights to kt-kernel optimized format.
        
        Args:
            model_path: Path to source model
            output_path: Path to save converted weights
            method: Quantization method
            numa_nodes: Number of NUMA nodes
            
        Returns:
            Path to converted weights
        """
        logger.info(f"Converting model weights to kt-kernel format: {method}")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Try to use kt-kernel conversion script
            from kt_kernel.cli.commands.quant import quantize_model
            
            quantize_model(
                model_path=model_path,
                output_path=str(output_path),
                method=method,
                numa_nodes=numa_nodes,
            )
        except ImportError:
            logger.warning("kt-kernel quantization not available, copying weights directly")
            
            # Fallback: just copy safetensors files
            import shutil
            model_path = Path(model_path)
            
            for f in model_path.glob("*.safetensors"):
                shutil.copy(f, output_path)
        
        return str(output_path)

    def clear_cache(self):
        """Clear model cache."""
        self._cached_models.clear()
        self._cached_processors.clear()
        logger.info("Model cache cleared")


def load_xoron_model(
    model_path: str,
    **kwargs,
) -> tuple[XoronForCausalLM, XoronMultimodalProcessor]:
    """
    Convenience function to load Xoron model.
    
    Args:
        model_path: HuggingFace model ID or local path
        **kwargs: Arguments passed to XoronModelLoader.load()
        
    Returns:
        Tuple of (model, processor)
    """
    loader = XoronModelLoader()
    return loader.load(model_path, **kwargs)


def get_xoron_sglang_config(
    model_path: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Get SGLang configuration for Xoron model.
    
    Args:
        model_path: HuggingFace model ID or local path
        **kwargs: Arguments passed to XoronModelLoader.load_for_sglang()
        
    Returns:
        SGLang configuration dictionary
    """
    loader = XoronModelLoader()
    return loader.load_for_sglang(model_path, **kwargs)
