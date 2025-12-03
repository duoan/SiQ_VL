"""
Processor class for SiQ-VL (Siglip2 + Qwen2).
"""

from typing import Any

import torch
from torchmetrics.utilities import rank_zero_info, rank_zero_warn
from transformers import (
    BatchEncoding,
    ProcessorMixin,
    Qwen2TokenizerFast,
    SiglipImageProcessor,
)


class SiQ_VLProcessor(ProcessorMixin):
    # Attributes required by ProcessorMixin for save_pretrained/from_pretrained to work
    attributes = ["image_processor", "tokenizer"]  # noqa: RUF012
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        image_processor: SiglipImageProcessor,
        tokenizer: Qwen2TokenizerFast,
        *,
        image_size: int = 384,
        patch_size: int = 14,
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
        self.image_size = image_size
        self.patch_size = patch_size

        # Calculate the number of image tokens dynamically
        self.num_image_tokens = self._calculate_num_image_tokens()

        # Overwrite the default text-only template with our multimodal-ready one.
        self._set_default_vlm_template()

        self._assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")

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

        # Calculate number of patches per dimension
        patches_per_dim = self.image_size // self.patch_size

        # After pixel shuffle, reduce by the shuffle factor
        reduced_patches_per_dim = patches_per_dim // self.pixel_shuffle_factor

        # Total number of tokens
        num_tokens = reduced_patches_per_dim**2

        rank_zero_info(
            f"Image tokenization: {self.image_size}x{self.image_size} image, "
            f"patch_size={self.patch_size}, "
            f"patches={patches_per_dim}x{patches_per_dim}={patches_per_dim**2}, "
            f"after pixel_shuffle(factor={self.pixel_shuffle_factor}): "
            f"{reduced_patches_per_dim}x{reduced_patches_per_dim}={num_tokens} tokens"
        )

        return num_tokens

    def _get_image_placeholder(self) -> str:
        """
        Generate the image placeholder string with vision tokens.

        Returns:
            String containing <|vision_start|><|image_pad|>...<|vision_end|>
        """
        image_pad_tokens = "<|image_pad|>" * self.num_image_tokens
        return f"<|vision_start|>{image_pad_tokens}<|vision_end|>"

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
        image_placeholder = self._get_image_placeholder()

        self.tokenizer.chat_template = f"""{{%- for message in messages %}}{{{{- '<|im_start|>' + message['role'] + '\\n' }}}}{{%- if message['content'] is string %}}{{{{- message['content'] }}}}{{%- else %}}{{%- for item in message['content'] %}}{{%- if item['type'] == 'text' %}}{{{{- item['text'] }}}}{{%- elif item['type'] == 'image' or item['type'] == 'image_url' %}}{{{{- '{image_placeholder}' }}}}{{%- endif %}}{{%- endfor %}}{{%- endif %}}{{{{- '<|im_end|>\\n' }}}}{{%- endfor %}}{{%- if add_generation_prompt %}}{{{{- '<|im_start|>assistant\\n' }}}}{{%- endif %}}"""

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
        """
        if not isinstance(batch, list) or not batch or not isinstance(batch[0], (tuple, list)) or len(batch[0]) != 3:
            raise ValueError(
                "SiQ_VLProcessor.__call__ expects `batch` to be a list of (image, question, answer_or_none). "
                f"Got type={type(batch)} with first element="
                f"{type(batch[0]) if isinstance(batch, list) and batch else None}."
            )

        images, questions, answers = zip(*batch, strict=False)

        # 1. Encode images
        image_outputs = self.image_processor(list(images), return_tensors=return_tensors)
        pixel_values = image_outputs["pixel_values"]

        # 2. Build ChatML-style messages per sample (so tokenization stays consistent with Qwen)
        msg_batch = []
        for q, a in zip(questions, answers, strict=False):
            q = q if q is not None else "Describe this image."
            if a is None:
                # Generation: only user turn
                msgs = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q},
                        ],
                    }
                ]
            else:
                # Supervised: user + assistant
                msgs = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q},
                        ],
                    },
                    {"role": "assistant", "content": a},
                ]
            msg_batch.append(msgs)

        # 3. Apply chat_template and tokenize (same behavior as Qwen chat)
        formatted_texts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=(ans is None),
            )
            for msgs, ans in zip(msg_batch, answers, strict=False)
        ]

        text_outputs = self.tokenizer(
            formatted_texts,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        data = dict(text_outputs)
        data["pixel_values"] = pixel_values

        # 4. If we have answers → build labels
        if any(a is not None for a in answers):
            input_ids = data["input_ids"]
            labels = torch.full_like(input_ids, fill_value=-100)

            tok = self.tokenizer
            im_start = tok.convert_tokens_to_ids("<|im_start|>")
            im_end = tok.convert_tokens_to_ids("<|im_end|>")
            assistant_id = tok.encode("assistant", add_special_tokens=False)[0]
            image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")

            for i in range(input_ids.size(0)):
                if answers[i] is None:
                    continue

                seq = input_ids[i]
                ids = seq.tolist()
                try:
                    for j, x in enumerate(ids):
                        if x == im_start and j + 1 < len(ids) and ids[j + 1] == assistant_id:
                            # j     : <|im_start|>
                            # j+1   : assistant
                            # j+2   : '\n'
                            start = j + 3
                            try:
                                end = ids.index(im_end, start)
                            except ValueError:
                                end = len(ids)
                            labels[i, start:end] = seq[start:end]
                            break
                except Exception as ex:  # pragma: no cover - debug only
                    rank_zero_warn(f"Warning: Failed to mask labels for sample {i}: {ex}")

                # Never predict image tokens
                labels[i, seq == image_token_id] = -100

            data["labels"] = labels

        return BatchEncoding(data=data, tensor_type=return_tensors)

    def batch_decode(self, sequences, assistant_only: bool = True, *args, **kwargs):
        """Helper to decode token IDs back to string."""
        if assistant_only:
            # Process each sequence in the batch to extract assistant-only parts
            filtered_sequences = []
            for seq in sequences:
                seq_tensor = seq if isinstance(seq, torch.Tensor) else torch.tensor(seq)

                # Find assistant token and extract from there
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
        if assistant_only:
            start_index = (output_ids == self._assistant_token_id).nonzero(as_tuple=True)[0] + 2
            output_ids = output_ids[start_index:]
        return self.tokenizer.decode(output_ids, *args, **kwargs)

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
    image_processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # 2. Initialize your Custom Processor
    processor = SiQ_VLProcessor(image_processor, tokenizer, image_size=224, patch_size=14, pixel_shuffle_factor=2)

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
    print(f"Decoded Prompt: \n{processor.decode(inputs.input_ids[0])}")

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
    print(f"Decoded Prompt: \n{processor.decode(inputs.input_ids[0])}")
