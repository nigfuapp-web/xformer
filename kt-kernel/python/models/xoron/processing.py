# Xoron Multimodal Processor
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal processor for Xoron model.

This module provides unified preprocessing for all modalities:
- Text tokenization
- Image preprocessing
- Video preprocessing
- Audio preprocessing
"""

import logging
from typing import Optional, Dict, List, Union, Any
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)


class XoronMultimodalProcessor:
    """
    Unified multimodal processor for Xoron model.
    
    Handles preprocessing for:
    - Text: Tokenization with Qwen2.5 tokenizer
    - Images: SigLIP/CLIP preprocessing with multi-scale support
    - Videos: Frame extraction and preprocessing
    - Audio: Waveform loading and normalization
    
    Usage:
        processor = XoronMultimodalProcessor.from_pretrained("your-repo/xoron-model")
        
        # Process multimodal inputs
        inputs = processor(
            text="Describe this image",
            images=["image.jpg"],
            videos=["video.mp4"],
            audio=["speech.wav"],
        )
        
        # Decode outputs
        text = processor.decode(outputs.logits.argmax(-1))
    """
    
    # Special tokens for multimodal content
    IMAGE_TOKEN = "<|image|>"
    VIDEO_TOKEN = "<|video|>"
    AUDIO_TOKEN = "<|audio|>"
    IMAGE_START_TOKEN = "<|image_start|>"
    IMAGE_END_TOKEN = "<|image_end|>"
    VIDEO_START_TOKEN = "<|video_start|>"
    VIDEO_END_TOKEN = "<|video_end|>"
    AUDIO_START_TOKEN = "<|audio_start|>"
    AUDIO_END_TOKEN = "<|audio_end|>"
    
    def __init__(
        self,
        tokenizer=None,
        image_processor=None,
        audio_sample_rate: int = 16000,
        video_max_frames: int = 24,
        image_size: int = 384,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_sample_rate = audio_sample_rate
        self.video_max_frames = video_max_frames
        self.image_size = image_size
        self._kwargs = kwargs
        
        # Add special tokens to tokenizer if needed
        if self.tokenizer is not None:
            self._add_special_tokens()

    def _add_special_tokens(self):
        """Add multimodal special tokens to tokenizer."""
        special_tokens = [
            self.IMAGE_TOKEN,
            self.VIDEO_TOKEN,
            self.AUDIO_TOKEN,
            self.IMAGE_START_TOKEN,
            self.IMAGE_END_TOKEN,
            self.VIDEO_START_TOKEN,
            self.VIDEO_END_TOKEN,
            self.AUDIO_START_TOKEN,
            self.AUDIO_END_TOKEN,
        ]
        
        # Check which tokens need to be added
        existing_tokens = set(self.tokenizer.get_vocab().keys())
        new_tokens = [t for t in special_tokens if t not in existing_tokens]
        
        if new_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            logger.info(f"Added {len(new_tokens)} special tokens to tokenizer")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer_name: str = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "XoronMultimodalProcessor":
        """
        Load processor from pretrained model.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            tokenizer_name: Override tokenizer name
            trust_remote_code: Trust remote code
            **kwargs: Additional arguments
            
        Returns:
            XoronMultimodalProcessor instance
        """
        from transformers import AutoTokenizer
        
        model_path = Path(model_path)
        
        # Load configuration to get tokenizer name
        config_path = model_path / "config.json" if model_path.exists() else None
        
        if tokenizer_name is None:
            if config_path and config_path.exists():
                import json
                with open(config_path) as f:
                    config = json.load(f)
                tokenizer_name = config.get("tokenizer_name", "Qwen/Qwen2.5-1.5B")
            else:
                tokenizer_name = "Qwen/Qwen2.5-1.5B"
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {tokenizer_name}, using default: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-1.5B",
                trust_remote_code=trust_remote_code,
            )
        
        # Load image processor
        image_processor = None
        try:
            from transformers import SiglipImageProcessor
            image_processor = SiglipImageProcessor.from_pretrained(
                "google/siglip-so400m-patch14-384"
            )
        except Exception as e:
            logger.warning(f"Could not load image processor: {e}")
        
        return cls(
            tokenizer=tokenizer,
            image_processor=image_processor,
            **kwargs,
        )

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[str, List[str], "PIL.Image.Image", List["PIL.Image.Image"]]] = None,
        videos: Optional[Union[str, List[str]]] = None,
        audio: Optional[Union[str, List[str], np.ndarray, List[np.ndarray]]] = None,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 2048,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process multimodal inputs.
        
        Args:
            text: Input text or list of texts
            images: Image paths, PIL images, or list thereof
            videos: Video paths or list thereof
            audio: Audio paths, numpy arrays, or list thereof
            return_tensors: Return type ("pt" for PyTorch)
            padding: Whether to pad inputs
            truncation: Whether to truncate inputs
            max_length: Maximum sequence length
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with processed inputs
        """
        batch_inputs = {}
        
        # Process text
        if text is not None:
            text_inputs = self._process_text(
                text,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
            )
            batch_inputs.update(text_inputs)
        
        # Process images
        if images is not None:
            image_inputs = self._process_images(images, return_tensors=return_tensors)
            batch_inputs.update(image_inputs)
        
        # Process videos
        if videos is not None:
            video_inputs = self._process_videos(videos, return_tensors=return_tensors)
            batch_inputs.update(video_inputs)
        
        # Process audio
        if audio is not None:
            audio_inputs = self._process_audio(audio, return_tensors=return_tensors)
            batch_inputs.update(audio_inputs)
        
        return batch_inputs

    def _process_text(
        self,
        text: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 2048,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Process text inputs."""
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def _process_images(
        self,
        images: Union[str, List[str], Any],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Process image inputs."""
        from PIL import Image
        
        if not isinstance(images, list):
            images = [images]
        
        # Load images
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Process with image processor
        if self.image_processor is not None:
            processed = self.image_processor(
                images=pil_images,
                return_tensors=return_tensors,
            )
            return {"pixel_values": processed["pixel_values"]}
        else:
            # Fallback: basic preprocessing
            import torchvision.transforms as T
            
            transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            
            tensors = torch.stack([transform(img) for img in pil_images])
            return {"pixel_values": tensors}

    def _process_videos(
        self,
        videos: Union[str, List[str]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Process video inputs."""
        if not isinstance(videos, list):
            videos = [videos]
        
        all_frames = []
        
        for video_path in videos:
            frames = self._extract_video_frames(video_path)
            all_frames.append(frames)
        
        # Stack all videos
        video_tensor = torch.stack(all_frames)
        
        return {"video_frames": video_tensor}

    def _extract_video_frames(self, video_path: str) -> torch.Tensor:
        """Extract frames from video file."""
        try:
            import cv2
            from PIL import Image
            import torchvision.transforms as T
        except ImportError:
            raise ImportError("cv2 and torchvision are required for video processing")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        if total_frames <= self.video_max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.video_max_frames, dtype=int)
        
        # Transform
        transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = transform(frame)
            frames.append(frame)
        
        cap.release()
        
        # Pad if needed
        while len(frames) < self.video_max_frames:
            frames.append(torch.zeros_like(frames[0]))
        
        return torch.stack(frames[:self.video_max_frames])

    def _process_audio(
        self,
        audio: Union[str, List[str], np.ndarray, List[np.ndarray]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Process audio inputs."""
        if not isinstance(audio, list):
            audio = [audio]
        
        waveforms = []
        
        for aud in audio:
            if isinstance(aud, str):
                waveform = self._load_audio(aud)
            elif isinstance(aud, np.ndarray):
                waveform = torch.from_numpy(aud).float()
            elif isinstance(aud, torch.Tensor):
                waveform = aud.float()
            else:
                raise ValueError(f"Unsupported audio type: {type(aud)}")
            
            # Normalize
            waveform = waveform / (waveform.abs().max() + 1e-8)
            waveforms.append(waveform)
        
        # Pad to same length
        max_len = max(w.shape[-1] for w in waveforms)
        padded = []
        
        for w in waveforms:
            if w.dim() == 1:
                w = w.unsqueeze(0)
            
            if w.shape[-1] < max_len:
                padding = torch.zeros(w.shape[0], max_len - w.shape[-1])
                w = torch.cat([w, padding], dim=-1)
            
            padded.append(w)
        
        audio_tensor = torch.stack([p.squeeze(0) for p in padded])
        
        return {"audio_features": audio_tensor}

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and resample if needed."""
        try:
            import torchaudio
        except ImportError:
            # Fallback to librosa
            try:
                import librosa
                waveform, sr = librosa.load(audio_path, sr=self.audio_sample_rate, mono=True)
                return torch.from_numpy(waveform)
            except ImportError:
                raise ImportError("torchaudio or librosa is required for audio processing")
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.audio_sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Skip special tokens in output
            **kwargs: Additional arguments
            
        Returns:
            Decoded text string(s)
        """
        if token_ids.dim() == 1:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        """Batch decode token IDs to text."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        return self.tokenizer.bos_token_id

    def save_pretrained(self, save_directory: str):
        """Save processor to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(save_directory))
        
        # Save image processor
        if self.image_processor is not None:
            self.image_processor.save_pretrained(str(save_directory))
        
        # Save processor config
        import json
        config = {
            "audio_sample_rate": self.audio_sample_rate,
            "video_max_frames": self.video_max_frames,
            "image_size": self.image_size,
        }
        
        with open(save_directory / "processor_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Processor saved to {save_directory}")
