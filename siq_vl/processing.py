# coding=utf-8
"""
Processor class for SiQ-VL (Siglip2 + Qwen2).
"""

from typing import Any, Dict, List, Optional, Union

from transformers import (
    BatchEncoding,
    ProcessorMixin,
    Qwen2TokenizerFast,
    SiglipImageProcessor,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SiQ_VLProcessor(ProcessorMixin):
    # Attributes required by ProcessorMixin for save_pretrained/from_pretrained to work
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        image_processor: SiglipImageProcessor,
        tokenizer: Qwen2TokenizerFast,
        pixel_shuffle_factor: int = 3,
    ):
        """
        Initializes the SiQ_VLProcessor.

        Args:
            image_processor: The SiglipImageProcessor (for visual inputs).
            tokenizer: The Qwen2TokenizerFast (for text inputs).
            pixel_shuffle_factor: The pixel shuffle factor used in the projector (default: 3).
        """
        super().__init__(image_processor, tokenizer)
        self.image_processor: SiglipImageProcessor = image_processor
        self.tokenizer: Qwen2TokenizerFast = tokenizer
        self.pixel_shuffle_factor = pixel_shuffle_factor

        # Calculate the number of image tokens dynamically
        self.num_image_tokens = self._calculate_num_image_tokens()

        # Overwrite the default text-only template with our multimodal-ready one.
        self._set_default_vlm_template()

    def _calculate_num_image_tokens(self) -> int:
        """
        Calculate the number of image placeholder tokens needed.
        
        Formula:
        - SigLIP image size: 384x384
        - Patch size: 14x14
        - Number of patches: (384/14)^2 = 27^2 = 729
        - After pixel shuffle (factor=3): (27/3)^2 = 9^2 = 81
        
        Returns:
            Number of image tokens after pixel shuffle
        """
        # Get image size from image processor
        if hasattr(self.image_processor, 'size'):
            if isinstance(self.image_processor.size, dict):
                image_size = self.image_processor.size.get("height", 384)
            else:
                image_size = self.image_processor.size
        else:
            image_size = 384  # Default for SigLIP SO400M
        
        # Get patch size from image processor config
        if hasattr(self.image_processor, 'patch_size'):
            patch_size = self.image_processor.patch_size
        elif hasattr(self.image_processor, 'config') and hasattr(self.image_processor.config, 'patch_size'):
            patch_size = self.image_processor.config.patch_size
        else:
            patch_size = 14  # Default for SigLIP SO400M
        
        # Calculate number of patches per dimension
        patches_per_dim = image_size // patch_size
        
        # After pixel shuffle, reduce by the shuffle factor
        reduced_patches_per_dim = patches_per_dim // self.pixel_shuffle_factor
        
        # Total number of tokens
        num_tokens = reduced_patches_per_dim ** 2
        
        logger.info(
            f"Image tokenization: {image_size}x{image_size} image, "
            f"patch_size={patch_size}, "
            f"patches={patches_per_dim}x{patches_per_dim}={patches_per_dim**2}, "
            f"after pixel_shuffle(factor={self.pixel_shuffle_factor}): "
            f"{reduced_patches_per_dim}x{reduced_patches_per_dim}={num_tokens} tokens"
        )
        
        return num_tokens

    def _set_default_vlm_template(self):
        """
        Injects a Jinja2 template into the tokenizer.

        Logic:
        1. Iterates over 'messages'.
        2. Handles standard text content.
        3. If content is a list (multimodal), it checks 'type'.
        4. If type is 'image', it inserts: <|vision_start|><|image_pad|>...(N times)...<|vision_end|>
        
        The number of <|image_pad|> tokens is calculated dynamically based on:
        - Image size and patch size from the image processor
        - Pixel shuffle factor from the projector
        """
        # Generate N <|image_pad|> tokens dynamically
        image_pad_tokens = '<|image_pad|>' * self.num_image_tokens
        image_placeholder = f'<|vision_start|>{image_pad_tokens}<|vision_end|>'
        
        self.tokenizer.chat_template = f"""{{%- for message in messages %}}{{{{- '<|im_start|>' + message['role'] + '\\n' }}}}{{%- if message['content'] is string %}}{{{{- message['content'] }}}}{{%- else %}}{{%- for item in message['content'] %}}{{%- if item['type'] == 'text' %}}{{{{- item['text'] }}}}{{%- elif item['type'] == 'image' or item['type'] == 'image_url' %}}{{{{- '{image_placeholder}' }}}}{{%- endif %}}{{%- endfor %}}{{%- endif %}}{{{{- '<|im_end|>\\n' }}}}{{%- endfor %}}{{%- if add_generation_prompt %}}{{{{- '<|im_start|>assistant\\n' }}}}{{%- endif %}}"""

    def __call__(  # type: ignore
        self,
        text: Union[str, List[str], List[Dict]] = None,  # type: ignore
        images: Union[Any, List[Any]] = None,
        return_tensors: Optional[str] = "pt",
        padding: Union[bool, str] = "longest",
        truncation: bool = True,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main entry point for processing inputs.

        Args:
            text: Can be a raw string, or a List of Dicts (ChatML format).
                  e.g., [{"role": "user", "content": [...]}]
            images: PIL Image object, or a list of PIL Images.
            return_tensors: "pt" for PyTorch, "np" for NumPy.
            padding: Padding strategy (default: "longest").
            truncation: Whether to truncate sequences.

        Returns:
            BatchEncoding: A dictionary containing 'input_ids', 'attention_mask',
                           and 'pixel_values'.
        """

        # 1. Process Images (if provided)
        pixel_values = None
        if images is not None:
            # SigLIP processor returns {'pixel_values': tensor}
            image_outputs = self.image_processor(images, return_tensors=return_tensors)
            pixel_values = image_outputs["pixel_values"]

        # 2. Process Text
        if text is not None:
            # Case A: Input is ChatML format (List of Dicts)
            # We need to apply the template to convert List -> String first.
            if (
                isinstance(text, list)
                and len(text) > 0
                and isinstance(text[0], dict)
                and "role" in text[0]
            ):
                text = self.tokenizer.apply_chat_template(
                    text,  # type: ignore
                    tokenize=False,
                    add_generation_prompt=kwargs.get("add_generation_prompt", False),
                )
            # Case B: Input is already a String (or list of strings)
            # Now we perform the actual tokenization (String -> IDs)
            text_outputs = self.tokenizer(
                text,  # type: ignore
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )
        else:
            text_outputs = BatchEncoding()

        # 3. Merge Results
        data = dict(text_outputs)
        if pixel_values is not None:
            data["pixel_values"] = pixel_values

        return BatchEncoding(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """Helper to decode token IDs back to string."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Helper to decode a single sequence of token IDs."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values"]


__all__ = ["SiQ_VLProcessor"]

SiQ_VLProcessor.register_for_auto_class()

if __name__ == "__main__":
    from PIL import Image
    from transformers import AutoImageProcessor, AutoTokenizer

    # 1. Load basic components
    # Note: Use SigLIP 1 config if SigLIP 2 isn't explicitly on HF yet (they are compatible)
    image_processor = AutoImageProcessor.from_pretrained(
        "google/siglip-so400m-patch14-384"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # 2. Initialize your Custom Processor
    processor = SiQ_VLProcessor(image_processor, tokenizer)

    # 3. Prepare Data (OpenAI/ChatML Format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "ignored_path"},  # Placeholder for logic
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]

    # Dummy image for testing
    raw_image = Image.new("RGB", (384, 384), color="red")

    # 4. Process
    inputs = processor(
        text=messages,
        images=raw_image,
        return_tensors="pt",
        add_generation_prompt=True,  # Adds <|im_start|>assistant
    )

    # 5. Verify
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    print(f"Pixel Values shape: {inputs.pixel_values.shape}")
    print(f"Decoded Prompt: \n{processor.decode(inputs.input_ids[0])}")
