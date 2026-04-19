"""
Model Loader: Load Qwen3-VL-2B-Instruct model for inference.
Only supports this specific model variant.
"""

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage Qwen3-VL-2B-Instruct VLM model."""

    # Only support Qwen3-VL-2B-Instruct
    DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True,
        cache_dir: str = "models/cache"
    ):
        """
        Initialize model loader.

        Args:
            model_name: Ignored - only Qwen3-VL-2B-Instruct is supported
            device: "cuda" or "cpu"
            torch_dtype: "float16", "bfloat16", or "auto"
            use_flash_attention: Enable flash attention for faster inference
            cache_dir: Directory to cache downloaded models
        """
        if model_name != self.DEFAULT_MODEL:
            logger.warning(f"Only Qwen3-VL-2B-Instruct is supported. Ignoring model_name={model_name}, using {self.DEFAULT_MODEL}")

        self.model_name = self.DEFAULT_MODEL
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

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        # Load model
        load_kwargs = {
            "cache_dir": self.cache_dir,
            "device_map": "auto" if self.device == "cuda" else "cpu",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        }

        if self.dtype != "auto":
            load_kwargs["torch_dtype"] = self.dtype

        if self.use_flash_attention and self.device == "cuda":
            try:
                from flash_attn import flash_attn_available
                if flash_attn_available():
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                else:
                    logger.warning("FlashAttention2 not available, using default attention")
            except ImportError:
                logger.warning("FlashAttention2 not installed. Install: pip install flash-attn --no-build-isolation")

        # Load model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
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
        repetition_penalty: float = 1.5,
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
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            enable_thinking: Enable thinking mode (Qwen3.x)
            preserve_thinking: Preserve thinking from history (Qwen3.x)
            **kwargs: Additional generation kwargs

        Returns:
            Generated text response
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Apply chat template with thinking mode
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={
                "enable_thinking": enable_thinking,
                "preserve_thinking": preserve_thinking
            }
        )

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

        # Use deterministic decoding for very low temperature to avoid runaway lists.
        do_sample = temperature is not None and temperature > 0.2

        # Generation kwargs (Qwen3-VL compatible)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
        }

        if do_sample:
            gen_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
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
