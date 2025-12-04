"""
Processor class for SiQ-VL (Siglip2 + Qwen2).
"""

import json
import math
import os
from typing import Any

from einops import rearrange
from PIL import Image
import torch
from torchmetrics.utilities import rank_zero_warn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, resize
from transformers import (
    AutoImageProcessor,
    BaseImageProcessor,
    BatchEncoding,
    BatchFeature,
    ProcessorMixin,
    Qwen2TokenizerFast,
)
from transformers.image_utils import ImageInput
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

DEFAULT_SYSTEM_PROMPT = (
    "You are SiQ-VL, an advanced vision-language assistant. "
    "You are built upon the Qwen2 language model and the SigLIP vision encoder. "
    "You can perceive, understand, and analyze images provided by the user. "
    "When answering questions about images, provide detailed, accurate, and helpful responses. "
    "If the image details are unclear, admit it rather than hallucinating."
)


class DynamicResize(torch.nn.Module):
    """
    Resize so that:
      * the longer side ≤ `max_side_len` **and** is divisible by `patch_size`
      * the shorter side keeps aspect ratio and is also divisible by `patch_size`
    Optionally forbids up-scaling.

    Works on PIL Images, (C, H, W) tensors, or (B, C, H, W) tensors.
    Returns the same type it receives.

    Copy from https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/data/custom_transforms.py
    """

    def __init__(
        self,
        tile_size: int,
        max_side_len: int,
        resize_to_max_side_len: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ) -> None:
        super().__init__()
        self.t = int(tile_size)
        self.m = int(max_side_len)
        self.interpolation = interpolation
        print(f"Resize to max side len: {resize_to_max_side_len}")
        self.resize_to_max_side_len = resize_to_max_side_len

    # ------------------------------------------------------------
    def _get_new_hw(self, h: int, w: int) -> tuple[int, int]:
        """Compute target (h, w) divisible by patch_size."""
        long, short = (w, h) if w >= h else (h, w)

        # 1) upscale long side
        target_long = self.m if self.resize_to_max_side_len else min(self.m, math.ceil(long / self.t) * self.t)

        # 2) scale factor
        scale = target_long / long

        # 3) compute short side with ceil → never undershoot
        target_short = math.ceil(short * scale / self.t) * self.t
        target_short = max(target_short, self.t)  # just in case

        return (target_short, target_long) if w >= h else (target_long, target_short)

    # ------------------------------------------------------------
    def forward(self, img: Image.Image | torch.Tensor):
        if isinstance(img, Image.Image):
            w, h = img.size
            new_h, new_w = self._get_new_hw(h, w)
            return resize(img, [new_h, new_w], interpolation=self.interpolation)

        if not torch.is_tensor(img):
            raise TypeError(f"DynamicResize expects a PIL Image or a torch.Tensor; got {type(img)}")

        # tensor path ---------------------------------------------------------
        batched = img.ndim == 4
        if img.ndim not in (3, 4):
            raise ValueError(f"Tensor input must have shape (C,H,W) or (B,C,H,W); got {img.shape}")

        # operate batch-wise
        imgs = img if batched else img.unsqueeze(0)
        _, _, h, w = imgs.shape
        new_h, new_w = self._get_new_hw(h, w)
        out = resize(imgs, [new_h, new_w], interpolation=self.interpolation)

        return out if batched else out.squeeze(0)


class SplitImage(torch.nn.Module):
    """Split (B, C, H, W) image tensor into square tiles.

    Returns:
        tiles: (B·n_h·n_w, C, tile_size, tile_size)
        grid:    (n_h, n_w)  - number of tiles along H and W
    """

    def __init__(self, tile_size: int) -> None:
        super().__init__()
        self.t = tile_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.ndim == 3:  # add batch dim if missing
            x = x.unsqueeze(0)

        b, c, h, w = x.shape
        if h % self.t or w % self.t:
            raise ValueError(f"Image size {(h, w)} not divisible by tile_size {self.t}")

        n_h, n_w = h // self.t, w // self.t
        tiles = rearrange(x, "b c (nh ph) (nw pw) -> (b nh nw) c ph pw", ph=self.t, pw=self.t)
        return tiles, (n_h, n_w)


class GlobalAndSplitImages(torch.nn.Module):
    def __init__(self, tile_size: int):
        super().__init__()
        self.t = tile_size
        self.splitter = SplitImage(tile_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.ndim == 3:
            x = x.unsqueeze(0)

        tiles, grid = self.splitter(x)

        if grid == (1, 1):
            return tiles, grid  # Dont add global tile if there is only one tile

        global_tile = resize(x, [self.t, self.t])
        return torch.cat([global_tile, tiles], dim=0), grid


def get_image_processor(tile_size: int, max_img_size: int, resize_to_max_side_len: bool = False):
    return transforms.Compose(
        [
            DynamicResize(tile_size, max_img_size, resize_to_max_side_len),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD),
            GlobalAndSplitImages(tile_size),
        ]
    )


class SiQ_VLImageProcessor(BaseImageProcessor):
    def __init__(self, vit_image_size: int = 224, **kwargs):
        super().__init__(**kwargs)
        self.vit_image_size = vit_image_size
        self.image_processor = get_image_processor(vit_image_size, vit_image_size * 4)

    def preprocess(
        self, images: ImageInput, return_tensors: str | torch.TensorType | None = None, **kwargs
    ) -> torch.Tensor:
        processed_pixel_values = []
        num_tiles_per_image = []

        for img in images:
            # Load path if string
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                if img.mode != "RGB":
                    img = img.convert("RGB")
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

            # Apply Tile Transform -> (N_tiles, 3, H, W)
            tiles, _grid = self.image_processor(img)

            processed_pixel_values.append(tiles)
            num_tiles_per_image.append(tiles.shape[0])

        # Concatenate all tiles from all images into a single batch tensor
        # Shape: (Total_Tiles_In_Batch, 3, H, W)
        pixel_values = torch.cat(processed_pixel_values, dim=0)
        num_tiles_per_image = torch.tensor(num_tiles_per_image, dtype=torch.long)
        return BatchFeature(
            data={"pixel_values": pixel_values, "num_tiles_per_image": num_tiles_per_image}, tensor_type=return_tensors
        )

    def to_dict(self):
        """Override to exclude non-serializable image_processor (Compose) attribute."""
        output = super().to_dict()
        # Remove the Compose object which is not JSON serializable
        # It will be reconstructed from vit_image_size during __init__
        output.pop("image_processor", None)
        return output


# Register the custom image processor so it can be found by ProcessorMixin
SiQ_VLImageProcessor.register_for_auto_class()
AutoImageProcessor.register(SiQ_VLImageProcessor, SiQ_VLImageProcessor)


class SiQ_VLProcessor(ProcessorMixin):
    # Attributes required by ProcessorMixin for save_pretrained/from_pretrained to work
    attributes = ["image_processor", "tokenizer"]  # noqa: RUF012
    image_processor_class = "SiQ_VLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    image_pad_token = "<|image_pad|>"
    vision_start_token = "<|vision_start|>"
    vision_end_token = "<|vision_end|>"
    assistant_token = "assistant"

    def check_argument_for_proper_class(self, argument_name: str, argument):
        """Override to handle custom SiQ_VLImageProcessor class."""
        # Handle custom image processor class
        if argument_name == "image_processor" and isinstance(argument, SiQ_VLImageProcessor):
            return SiQ_VLImageProcessor
        # Fall back to parent implementation for other arguments
        return super().check_argument_for_proper_class(argument_name, argument)

    def __init__(
        self,
        image_processor: SiQ_VLImageProcessor | Qwen2TokenizerFast | None = None,
        tokenizer: Qwen2TokenizerFast | None = None,
        *,
        vit_image_size: int = 224,  # same size as the Siglip image processor
        vit_patch_size: int = 16,
        pixel_shuffle_factor: int = 2,
        system_prompt: str | None = None,
    ):
        """
        Initializes the SiQ_VLProcessor.

        Args:
            image_processor: The SiQ_VLImageProcessor (for visual inputs). If None, will be created from vit_image_size.
                For backward compatibility, if a Qwen2TokenizerFast is passed here, it will be treated as tokenizer.
            tokenizer: The Qwen2TokenizerFast (for text inputs). Required if image_processor is None or is a tokenizer.
            vit_image_size: Image size for the vision encoder (default: 224).
            vit_patch_size: Patch size for the vision encoder (default: 16).
            pixel_shuffle_factor: The pixel shuffle factor used in the projector (default: 2).
            system_prompt: Custom system prompt (default: uses DEFAULT_SYSTEM_PROMPT).
        """
        # Handle backward compatibility: if first arg is a tokenizer, treat it as tokenizer
        if isinstance(image_processor, Qwen2TokenizerFast):
            tokenizer = image_processor
            image_processor = None

        if image_processor is None:
            if tokenizer is None:
                raise ValueError("Either image_processor or tokenizer must be provided")
            self.image_processor = SiQ_VLImageProcessor(vit_image_size=vit_image_size)
        else:
            self.image_processor = image_processor
        self.tokenizer: Qwen2TokenizerFast = tokenizer

        super().__init__(image_processor=self.image_processor, tokenizer=self.tokenizer)

        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.vit_image_size = vit_image_size
        self.vit_patch_size = vit_patch_size
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # Calculate the number of image tokens dynamically
        self.patches_per_dim = vit_image_size // vit_patch_size  # e.g. 384 // 14 = 27
        self.reduced_dim = self.patches_per_dim // pixel_shuffle_factor  # e.g. 27 // 4 = 6
        self.tokens_per_tile = self.reduced_dim**2  # e.g. 6^2 = 36

        self._im_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._assistant_token_id = self.tokenizer.convert_tokens_to_ids(self.assistant_token)
        self._image_pad_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad_token)

        self._set_chat_template()

    def _set_chat_template(self):
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )

    def _normalize_message_content(self, content):
        """Normalize message content to string format."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Extract text from list format: [{'type': 'text', 'text': '...'}]
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "".join(text_parts)
        return str(content)

    def __call__(  # type: ignore
        self,
        batch: list[tuple[Any, str, str | None]],
        return_tensors: str | None = "pt",
        padding: bool | str = "longest",
        truncation: bool = True,
        max_length: int | None = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main entry point: expects a list of (image, question, answer_or_none).

        - If answer is None → generation mode (adds generation prompt in chat_template)
        - If answer is string → supervised mode, will also return labels.

        Integrates Dynamic Tiling (AnyResize)
        """
        if not isinstance(batch, list) or not batch or not isinstance(batch[0], (tuple, list)) or len(batch[0]) != 3:
            raise ValueError(
                "SiQ_VLProcessor.__call__ expects `batch` to be a list of (image, question, answer_or_none). "
                f"Got type={type(batch)} with first element="
                f"{type(batch[0]) if isinstance(batch, list) and batch else None}."
            )

        images, questions, answers = zip(*batch, strict=False)

        # 1. Encode images
        # -------------------------------------------------------
        # 1. Image Processing (Tile / AnyRes Logic)
        # -------------------------------------------------------
        image_features = self.image_processor(images, return_tensors=return_tensors)
        pixel_values = image_features["pixel_values"]
        num_tiles_per_image = image_features["num_tiles_per_image"]

        # -------------------------------------------------------
        # 2. Text Processing (Dynamic Token Insertion)
        # -------------------------------------------------------
        msg_batch = []

        for q, a, n_tiles in zip(questions, answers, num_tiles_per_image, strict=False):
            q = q if q is not None else "Describe this image."
            total_image_tokens = n_tiles * self.tokens_per_tile

            # Construct placeholder: <|vision_start|><|image_pad|>...<|vision_end|>
            image_placeholder = (
                self.vision_start_token + self.image_pad_token * total_image_tokens + self.vision_end_token
            )

            user_content = q
            if "<image>" in user_content:
                user_content = user_content.replace("<image>", image_placeholder)
            else:
                if n_tiles > 0:
                    user_content = image_placeholder + "\n" + user_content

            msgs = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            if a is not None:
                msgs.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": a},
                        ],
                    }
                )
            print(msgs)
            msg_batch.append(msgs)

        # -------------------------------------------------------
        # 3. Tokenize
        # -------------------------------------------------------
        # Normalize message content to strings before applying template
        normalized_msg_batch = []
        for msgs in msg_batch:
            normalized_msgs = [{**msg, "content": self._normalize_message_content(msg["content"])} for msg in msgs]
            normalized_msg_batch.append(normalized_msgs)

        formatted_texts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=(ans is None),
            )
            for msgs, ans in zip(normalized_msg_batch, answers, strict=False)
        ]

        print(f"Formatted Texts: \n{formatted_texts[0]}")

        text_outputs = self.tokenizer(
            formatted_texts,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        data = dict(text_outputs)
        data["pixel_values"] = pixel_values
        # Optional: Pass num_tiles info to model if needed for advanced masking
        # data["image_sizes"] = torch.tensor(num_tiles_per_image)

        # -------------------------------------------------------
        # 4. Label Masking (Supervised Mode)
        # -------------------------------------------------------
        if any(a is not None for a in answers):
            input_ids = data["input_ids"]
            labels = torch.full_like(input_ids, fill_value=-100)

            for i in range(input_ids.size(0)):
                if answers[i] is None:
                    continue

                seq = input_ids[i]
                ids = seq.tolist()
                try:
                    for j, x in enumerate(ids):
                        if x == self._im_start_token_id and j + 1 < len(ids) and ids[j + 1] == self._assistant_token_id:
                            # j     : <|im_start|>
                            # j+1   : assistant
                            # j+2   : '\n'
                            start = j + 3
                            try:
                                end = ids.index(self._im_end_token_id, start)
                            except ValueError:
                                end = len(ids)
                            labels[i, start:end] = seq[start:end]
                            break
                except Exception as ex:  # pragma: no cover - debug only
                    rank_zero_warn(f"Warning: Failed to mask labels for sample {i}: {ex}")

                # Strict Masking: Never predict image tokens (safety net)
                labels[i, seq == self._image_pad_token_id] = -100

            data["labels"] = labels

        return BatchEncoding(data=data, tensor_type=return_tensors)

    def batch_decode(self, sequences, assistant_only: bool = True, *args, **kwargs):
        """Helper to decode token IDs back to string."""
        if assistant_only:
            # Process each sequence in the batch to extract assistant-only parts
            filtered_sequences = []
            for seq in sequences:
                seq_tensor = seq if isinstance(seq, torch.Tensor) else torch.tensor(seq)
                ids = seq_tensor.tolist()

                # Find <|im_start|> followed by assistant token (consistent with label masking)
                # Format: <|im_start|> (j), assistant (j+1), '\n' (j+2), response starts (j+3)
                start_index = None
                for j, x in enumerate(ids):
                    if x == self._im_start_token_id and j + 1 < len(ids) and ids[j + 1] == self._assistant_token_id:
                        # j     : <|im_start|>
                        # j+1   : assistant
                        # j+2   : '\n'
                        start_index = j + 3
                        break

                if start_index is not None:
                    filtered_seq = seq_tensor[start_index:]
                else:
                    # Fallback: try to find assistant token directly (for backward compatibility)
                    assistant_indices = (seq_tensor == self._assistant_token_id).nonzero(as_tuple=True)[0]
                    if len(assistant_indices) > 0:
                        start_index = assistant_indices[0] + 2
                        filtered_seq = seq_tensor[start_index:]
                    else:
                        filtered_seq = seq_tensor
                filtered_sequences.append(filtered_seq)

            sequences = filtered_sequences

        return self.tokenizer.batch_decode(sequences, *args, **kwargs)

    def decode(self, output_ids: torch.Tensor, assistant_only: bool = True, *args, **kwargs):
        """Helper to decode a single sequence of token IDs."""
        # Convert to tensor if it's a list or other type
        if not isinstance(output_ids, torch.Tensor):
            output_ids = torch.tensor(output_ids)

        if assistant_only:
            ids = output_ids.tolist()

            # Find <|im_start|> followed by assistant token (consistent with label masking)
            # Format: <|im_start|> (j), assistant (j+1), '\n' (j+2), response starts (j+3)
            start_index = None
            for j, x in enumerate(ids):
                if x == self._im_start_token_id and j + 1 < len(ids) and ids[j + 1] == self._assistant_token_id:
                    # j     : <|im_start|>
                    # j+1   : assistant
                    # j+2   : '\n'
                    start_index = j + 3
                    break

            if start_index is not None:
                output_ids = output_ids[start_index:]
            else:
                # Fallback: try to find assistant token directly (for backward compatibility)
                assistant_indices = (output_ids == self._assistant_token_id).nonzero(as_tuple=True)[0]
                if len(assistant_indices) > 0:
                    start_index = assistant_indices[0] + 2
                    output_ids = output_ids[start_index:]
        return self.tokenizer.decode(output_ids, *args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values"]

    def to_dict(self, legacy_serialization=True):
        output = super().to_dict(legacy_serialization=legacy_serialization)
        output.pop("image_processor", None)  # Remove if exists
        output["vit_image_size"] = self.vit_image_size
        output["vit_patch_size"] = self.vit_patch_size
        output["pixel_shuffle_factor"] = self.pixel_shuffle_factor
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load processor from pretrained, ensuring custom attributes are loaded from processor_config.json.
        """
        # Load processor_config.json first to get custom attributes
        from transformers.utils import PROCESSOR_NAME, cached_file

        processor_config_file = cached_file(
            pretrained_model_name_or_path,
            PROCESSOR_NAME,
            _raise_exceptions_for_missing_entries=False,
        )

        # Extract custom attributes from processor_config.json and pass as kwargs
        if processor_config_file and os.path.exists(processor_config_file):
            with open(processor_config_file, encoding="utf-8") as f:
                processor_config = json.load(f)

            # Pass custom attributes as kwargs if they exist in config and not already in kwargs
            if "vit_image_size" in processor_config and "vit_image_size" not in kwargs:
                kwargs["vit_image_size"] = processor_config["vit_image_size"]
            if "vit_patch_size" in processor_config and "vit_patch_size" not in kwargs:
                kwargs["vit_patch_size"] = processor_config["vit_patch_size"]
            if "pixel_shuffle_factor" in processor_config and "pixel_shuffle_factor" not in kwargs:
                kwargs["pixel_shuffle_factor"] = processor_config["pixel_shuffle_factor"]

        # Load using parent method with updated kwargs
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


__all__ = ["SiQ_VLProcessor"]

SiQ_VLProcessor.register_for_auto_class()

if __name__ == "__main__":
    from PIL import Image
    from transformers import AutoTokenizer

    # 1. Load basic components
    # Note: Use SigLIP 1 config if SigLIP 2 isn't explicitly on HF yet (they are compatible)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # 2. Initialize your Custom Processor
    processor = SiQ_VLProcessor(tokenizer, vit_image_size=224, vit_patch_size=14, pixel_shuffle_factor=2)

    # Dummy image for testing
    raw_image = Image.new("RGB", (384, 384), color="red")

    # 3. Test image + question + answer
    print("--- test image + question + answer ---")
    inputs = processor(
        batch=[(raw_image, "How many people are in the image?", "There are 2 people in the image.")],
        return_tensors="pt",
    )
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    print(f"Input IDs: {inputs.input_ids[0]}")
    print(f"Pixel Values shape: {inputs.pixel_values.shape}")
    print(f"Pixel Values: {inputs.pixel_values[0]}")
    print(f"Labels shape: {inputs.labels.shape}")
    print(f"Labels: {inputs.labels[0]}")
    print(f"Attention Mask shape: {inputs.attention_mask.shape}")
    print(f"Attention Mask: {inputs.attention_mask[0]}")
    print(f"Decoded Prompt: \n{processor.decode(inputs.input_ids[0], assistant_only=False)}")

    # -- test question only
    print("--- test image + question only ---")
    inputs = processor(batch=[(raw_image, "How are you?", None)], return_tensors="pt")
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    print(f"Pixel Values shape: {inputs.pixel_values.shape}")
    print(f"Decoded Prompt: \n{processor.decode(inputs.input_ids[0])}")

    # -- test image only
    print("--- test image only ---")
    inputs = processor(batch=[(raw_image, None, None)], return_tensors="pt")
    print(f"Pixel Values shape: {inputs.pixel_values.shape}")
    print(f"Decoded Prompt: \n{processor.decode(inputs.input_ids[0])}")

    # -- test multiple samples
    print("--- test multiple samples ---")
    inputs = processor(
        batch=[
            (raw_image, "What is the color of the main subject?", "The color of the main subject is red."),
            (raw_image, "What is the size of the main subject?", "The size of the main subject is 100x100 pixels."),
            (raw_image, "What is the position of the main subject?", "The position of the main subject is 100, 100."),
            (raw_image, "What is the texture of the main subject?", "The texture of the main subject is smooth."),
            (raw_image, "What is the shape of the main subject?", "The shape of the main subject is a rectangle."),
        ],
        return_tensors="pt",
    )
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    print(f"Pixel Values shape: {inputs.pixel_values.shape}")
    print(f"Labels shape: {inputs.labels.shape}")
    print(f"Labels: {inputs.labels[0]}")
    print(f"Attention Mask shape: {inputs.attention_mask.shape}")
    print(f"Attention Mask: {inputs.attention_mask[0]}")
    print(f"Decoded Prompt: \n{processor.decode(inputs.input_ids[0], skip_special_tokens=True)}")
