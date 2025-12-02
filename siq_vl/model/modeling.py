from PIL import Image
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    Cache,
    Qwen2ForCausalLM,
    Qwen2TokenizerFast,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.utils.generic import TransformersKwargs

from siq_vl.model.configuration import SiQ_VLConfig
from siq_vl.model.processing import SiQ_VLProcessor

logger = logging.get_logger(__name__)


class SiQ_VLMultiModalityProjector(nn.Module):
    def __init__(
        self,
        vision_hidden_size,
        vision_pixel_shuffle_factor,
        language_model_hidden_size,
    ):
        super().__init__()
        self.vision_pixel_shuffle_factor = vision_pixel_shuffle_factor
        input_dim = vision_hidden_size * (vision_pixel_shuffle_factor**2)
        self.proj = nn.Linear(input_dim, language_model_hidden_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def _pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        # Sequence length must be a perfect square for pixel shuffle
        if seq_root**2 != seq:
            raise ValueError(
                f"Sequence length {seq} is not a perfect square (sqrt={seq_root}). "
                f"Cannot apply pixel shuffle. Please check your vision model configuration."
            )
        # Sequence root must be divisible by scale factor
        if seq_root % self.vision_pixel_shuffle_factor != 0:
            # Find valid factors
            valid_factors = [f for f in range(1, seq_root + 1) if seq_root % f == 0]
            raise ValueError(
                f"seq_root {seq_root} is not divisible by scale factor {self.vision_pixel_shuffle_factor}. "
                f"Valid factors for seq_root {seq_root} are: {valid_factors}. "
                f"Please set pixel_shuffle_factor to one of these values."
            )

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.vision_pixel_shuffle_factor
        w_out = width // self.vision_pixel_shuffle_factor

        x = x.reshape(bsz, h_out, self.vision_pixel_shuffle_factor, w_out, self.vision_pixel_shuffle_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.vision_pixel_shuffle_factor**2)

        return x

    def forward(self, x):
        x = self._pixel_shuffle(x)
        x = self.proj(x)
        return x


class SiQ_VLModel(Qwen2ForCausalLM):
    """
    SiQ-VL model combining SigLIP vision encoder with Qwen language model.

    This model inherits from PreTrainedModel which provides the methods:
    - save_pretrained() and from_pretrained() for saving/loading
    - get_input_embeddings() and set_input_embeddings()
    - and a few others generic methods

    See the documentation for all the methods available.
    """

    config_class = SiQ_VLConfig
    model_type = "siq_vl"
    base_model_prefix = "siq_vl"

    # CRITICAL: Tell Trainer we don't accept loss kwargs
    # This ensures Trainer properly normalizes loss for gradient accumulation
    # See: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L4060-4064
    accepts_loss_kwargs = False

    def __init__(
        self,
        config: SiQ_VLConfig = None,
    ):
        """
        Initialize SiQ_VLModel.

        Args:
            config: SiQ_VLConfig instance. If None, will be created from other args.
            gradient_accumulation_steps: Gradient accumulation steps (for compatibility)
        """
        super().__init__(config)

        self.vision_model = SiglipVisionModel.from_pretrained(config.pretrained_vision_model_path)

        self.mm_projector = SiQ_VLMultiModalityProjector(
            self.config.vision_hidden_size,
            self.config.vision_pixel_shuffle_factor,
            self.config.language_model_hidden_size,  # type: ignore
        )
        if self.config.freeze_vision_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad_(False)
            self.vision_model.eval()
            # Only check for NaN if not on meta device
            for name, param in self.vision_model.named_parameters():
                if param.device.type != "meta" and torch.isnan(param).any():
                    logger.warning(f"NaN detected in vision model parameter: {name}")

        if self.config.freeze_language_model:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()

        # Store token IDs from config
        self.language_model_ignore_index = config.language_model_ignore_index
        self.language_model_image_token_id = config.language_model_image_token_id
        self.language_model_vision_start_id = config.language_model_vision_start_id
        self.language_model_vision_end_id = config.language_model_vision_end_id

        self.mm_projector.apply(self._init_mm_projector_weights)

        super().post_init()

    def _init_mm_projector_weights(self, m):
        """
        Custom initialization to align the multi-modality projector's output distribution
        with the language model's embedding distribution.
        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for SiQ-VL model that combines vision and language embeddings.
        Arguments:
            input_ids: The input text token ids (batch_size, sequence_length).
            pixel_values: The input vision features (batch_size, channels, height, width).
            attention_mask: The attention mask for the language model (batch_size, sequence_length).
            position_ids: The position ids for the language model (batch_size, sequence_length).
            past_key_values: The past key values from the language model.
            inputs_embeds: The input embeddings for the language model.
            labels: The target text token ids (batch_size, sequence_length).
            use_cache: Whether to use cache for the language model.
            cache_position: The cache position for the language model.
            logits_to_keep: The number of logits to keep for the language model.
            **kwargs: Additional arguments for the language model forward pass.
        """
        # Process vision inputs only on the first forward pass (when past_key_values is None)
        # During generation with KV cache, we only process new tokens, not images
        if past_key_values is None and pixel_values is not None:
            vision_outputs = self.vision_model(pixel_values)
            vision_features = vision_outputs.last_hidden_state
            vision_token_embeddings = self.mm_projector(vision_features)

            token_embeddings = self.get_input_embeddings()(input_ids)
            # find the positions of <|image_pad|> tokens
            image_token_positions = (input_ids == self.language_model_image_token_id).nonzero(as_tuple=True)

            if image_token_positions[0].numel() > 0:
                batch_size, num_image_tokens, hidden = vision_token_embeddings.shape
                expected = batch_size * num_image_tokens
                actual = image_token_positions[0].numel()
                if actual != expected:
                    raise ValueError(
                        f"Number of <|image_pad|> tokens ({actual}) "
                        f"!= batch_size * num_image_tokens from vision encoder ({expected}). "
                        "Please check processor image_size/patch_size/pixel_shuffle_factor "
                        "and vision_pixel_shuffle_factor in config."
                    )
                vision_token_embeddings_flat = vision_token_embeddings.reshape(expected, hidden)
                # replace the <|image_pad|> tokens with the vision token embeddings
                # Ensure dtype match to avoid RuntimeError
                vision_token_embeddings_flat = vision_token_embeddings_flat.to(dtype=token_embeddings.dtype)
                token_embeddings[image_token_positions] = vision_token_embeddings_flat
        else:
            # During generation with KV cache, only process new text tokens
            token_embeddings = self.get_input_embeddings()(input_ids)

        return super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=token_embeddings,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


__all__ = ["SiQ_VLModel"]

AutoModelForCausalLM.register(config_class=SiQ_VLConfig, model_class=SiQ_VLModel)

# test code
if __name__ == "__main__":
    config = SiQ_VLConfig(
        pretrained_vision_model_path="google/siglip2-base-patch16-224",
        pretrained_language_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        vision_hidden_size=768,
        vision_pixel_shuffle_factor=2,
        language_model_hidden_size=768,
        language_model_image_token_id=151655,
        language_model_vision_start_id=151652,
        language_model_vision_end_id=151653,
        language_model_ignore_index=-100,
    )

    model = SiQ_VLModel(config=config)
    processor = SiQ_VLProcessor(
        image_processor=SiglipImageProcessor.from_pretrained(config.pretrained_vision_model_path),
        tokenizer=Qwen2TokenizerFast.from_pretrained(config.pretrained_language_model_path),
        image_size=config.vision_image_size,
        patch_size=config.vision_patch_size,
        pixel_shuffle_factor=config.vision_pixel_shuffle_factor,
    )
    inputs = processor(
        batch=[
            (Image.open("image.png"), "Describe this image.", "The image shows a beautiful sunset."),
            (Image.open("image.png"), "How many people are in the image?", "There are 2 people in the image."),
        ],
        return_tensors="pt",
    )

    print("pixel_values min and max:", inputs.pixel_values.min(), inputs.pixel_values.max())
    print("pixel_values shape:", inputs.pixel_values.shape)

    outputs = model(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        attention_mask=inputs.attention_mask,
        labels=inputs.labels,
    )
    print(outputs)

    print("Generating...")
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            max_new_tokens=64,
        )

    print(processor.batch_decode(output_ids, assistant_only=True, skip_special_tokens=True))
