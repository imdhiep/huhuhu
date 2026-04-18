"""
Model Loader: Load Qwen2.5-VL or Qwen3-VL models for inference.
Supports both local and HuggingFace models.
"""

import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration
)
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage VLM models (Qwen2.5-VL / Qwen3-VL)."""

    # Model mappings
    QWEN2_MODELS = {
        "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "72b": "Qwen/Qwen2.5-VL-72B-Instruct"
    }

    QWEN3_MODELS = {
        "32b": "Qwen/Qwen3-VL-32B-Instruct",
        "30b-a3b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "3.6-35b-a3b": "Qwen/Qwen3.6-35B-A3B-Instruct"  # If available
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True,
        cache_dir: str = "models/cache"
    ):
        """
        Initialize model loader.

        Args:
            model_name: HuggingFace model ID or local path
            device: "cuda" or "cpu"
            torch_dtype: "float16", "bfloat16", or "auto"
            use_flash_attention: Enable flash attention for faster inference
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set dtype
        if torch_dtype == "float16":
            self.dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = "auto"

        self.use_flash_attention = use_flash_attention

        self.model = None
        self.processor = None

    def load_model(self):
        """Load the model and processor."""
        logger.info(f"Loading model: {self.model_name}")

        # Determine model class based on name
        if "qwen2" in self.model_name.lower() or "qwen2.5" in self.model_name.lower():
            model_class = Qwen2_5_VLForConditionalGeneration
        elif "qwen3" in self.model_name.lower():
            model_class = Qwen3VLForConditionalGeneration
        else:
            # Auto-detect
            model_class = AutoModelForVision2Seq

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )

        # Load model
        load_kwargs = {
            "cache_dir": self.cache_dir,
            "device_map": "auto" if self.device == "cuda" else "cpu",
        }

        if self.dtype != "auto":
            load_kwargs["torch_dtype"] = self.dtype

        if self.use_flash_attention and self.device == "cuda":
            # Enable flash attention if supported
            load_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = model_class.from_pretrained(
            self.model_name,
            **load_kwargs
        )

        logger.info(f"Model loaded successfully on {self.device}")
        return self.model, self.processor

    def generate(
        self,
        messages: list,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 20,
        presence_penalty: float = 1.5,
        enable_thinking: bool = True,
        preserve_thinking: bool = False,
        **kwargs
    ):
        """
        Generate response from model.

        Args:
            messages: List of conversation messages
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            presence_penalty: Presence penalty
            enable_thinking: Enable thinking mode (for Qwen3.x)
            preserve_thinking: Preserve thinking from history (Qwen3.x)
            **kwargs: Additional generation kwargs

        Returns:
            Generated text response
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={
                "enable_thinking": enable_thinking,
                "preserve_thinking": preserve_thinking
            } if "qwen3" in self.model_name.lower() else None
        )

        # Process inputs
        # For video: pass images directly to processor
        # Extract images from messages
        images = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        images.append(item["image"])

        inputs = self.processor(
            text=text,
            images=images if images else None,
            return_tensors="pt"
        ).to(self.device)

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
        }
        gen_kwargs.update(kwargs)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only new tokens
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]
        response = self.processor.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return response.strip()

    def to(self, device: str):
        """Move model to specified device."""
        if self.model is not None:
            self.model.to(device)
            self.device = device

    def clear_cache(self):
        """Clear model from GPU memory."""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()
