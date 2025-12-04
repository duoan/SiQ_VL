from typing import ClassVar

from PIL import Image
import torch
import torch.nn as nn
from torchmetrics.utilities.prints import rank_zero_info
from transformers import (
    AutoModelForCausalLM,
    Cache,
    GenerationMixin,
    PreTrainedModel,
    Qwen2ForCausalLM,
    Qwen2TokenizerFast,
    SiglipVisionModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.utils.generic import TransformersKwargs

from siq_vl.model.configuration import (
    SiQ_VLConfig,
    SiQ_VLProjectorConfig,
    SiQ_VLTextConfig,
    SiQ_VLVisionConfig,
    get_siq_vl_config,
)
from siq_vl.model.processing import SiQ_VLProcessor

logger = logging.get_logger(__name__)


class SiQ_VLProjector(PreTrainedModel):
    """
    SiQ-VL projector module.
    This module is used to project the vision features into the language model embedding space.
    It consists of a pixel shuffle operation, a linear projection and a MLP.
    The pixel shuffle operation is used to compress the vision features into a lower dimension.
    The linear projection is used to project the vision features into the language model embedding space.
    The MLP is used to further transform the language model embeddings.
    The output of the MLP is added to the input of the linear projection to form the final output.
    """

    config: SiQ_VLProjectorConfig = None

    def __init__(
        self,
        config: SiQ_VLProjectorConfig = None,
    ):
        super().__init__(config)
        self.config = config
        self.vision_pixel_shuffle_factor = config.vision_pixel_shuffle_factor
        input_dim = config.vision_hidden_size * (config.vision_pixel_shuffle_factor**2)
        self.mlp = nn.Linear(input_dim, config.text_hidden_size, bias=False)
        self.apply(self._init_weights)

        self.post_init()

    def _init_weights(self, module):
        """
        Custom initialization to align the multi-modality projector's output distribution
        with the language model's embedding distribution.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
        return x.reshape(bsz, h_out * w_out, embed_dim * self.vision_pixel_shuffle_factor**2)

    def forward(self, x):
        # 1. pixel shuffle: [batch_size, L, D] -> [batch_size, L/r^2, D*r^2]
        x = self._pixel_shuffle(x)
        # 2. Simple MLP
        return self.mlp(x)


class SiQ_VLVisionModel(SiglipVisionModel):
    """
    SiQ-VL vision model.
    """

    config: SiQ_VLVisionConfig = None

    def __init__(self, config: SiQ_VLVisionConfig = None):
        super().__init__(config)
        self.config = config


class SiQ_VLPreTrainedModel(PreTrainedModel):
    """
    SiQ-VL pre-trained model.
    """

    config: SiQ_VLConfig = None
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = ["SiQ_VLTextModel", "SiQ_VLVisionModel", "SiQ_VLProjector"]
    _skip_keys_device_placement: ClassVar[list[str]] = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True


class SiQ_VLTextModel(Qwen2ForCausalLM):
    config: SiQ_VLTextConfig = None

    def __init__(self, config: SiQ_VLTextConfig = None):
        super().__init__(config)
        self.config = config

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()


class SiQ_VLForCausalLM(SiQ_VLPreTrainedModel, GenerationMixin):
    def __init__(self, config: SiQ_VLConfig = None):
        super().__init__(config)
        self.config = config
        self.text_model = SiQ_VLTextModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.vision_model = SiQ_VLVisionModel(config.vision_config)
        self.projector = SiQ_VLProjector(config.projector_config)

        # Initialize weights and apply final processing
        self.post_init()

    def freez_text_model(self):
        for param in self.text_model.parameters():
            param.requires_grad_(False)
        self.text_model.eval()

    def unfreez_text_model(self):
        for param in self.text_model.parameters():
            param.requires_grad_(True)
        self.text_model.eval()

    def freez_vision_model(self):
        for param in self.vision_model.parameters():
            param.requires_grad_(False)
        self.vision_model.eval()

    def print_trainable_parameters(self):
        """
        Print the number of trainable parameters for each submodel and the total number of trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        vision_params = sum(p.numel() for p in self.vision_model.parameters())
        text_params = sum(p.numel() for p in self.text_model.parameters())
        projector_params = sum(p.numel() for p in self.projector.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        rank_zero_info(
            f"Total model parameters: {total_params:,} | "
            f"Vision model parameters: {vision_params:,} ({vision_params / total_params * 100:.2f}%) | "
            f"Text model parameters: {text_params:,} ({text_params / total_params * 100:.2f}%) | "
            f"Projector parameters: {projector_params:,} ({projector_params / total_params * 100:.2f}%) | "
            f"Trainable model parameters: {trainable_params:,}, ({trainable_params / total_params * 100:.2f}%)"
        )

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
            # 1. Vision Forward
            # pixel_values shape: (Total_Tiles, C, H, W)
            vision_outputs = self.vision_model(pixel_values)
            vision_features = vision_outputs.last_hidden_state

            # 2. Projector
            # vision_token_embeddings shape: (Total_Tiles, Tokens_Per_Tile, Text_Dim)
            vision_token_embeddings = self.projector(vision_features)

            # 3. Flatten All Vision Tokens
            # vision_token_embeddings_flat shape: (Total_Vision_Tokens, Text_Dim)
            vision_token_embeddings_flat = vision_token_embeddings.view(-1, vision_token_embeddings.size(-1))

            # 4. Text Embeddings
            token_embeddings = self.text_model.get_input_embeddings()(input_ids)

            # 5. Replacement
            # find the positions of <|image_pad|> tokens
            image_token_positions = (input_ids == self.config.image_token_index).nonzero(as_tuple=True)

            num_placeholders = image_token_positions[0].numel()
            num_vision_tokens = vision_token_embeddings_flat.shape[0]

            if num_placeholders != num_vision_tokens:
                raise ValueError(
                    f"Shape Mismatch! \n"
                    f"Text has {num_placeholders} image placeholders.\n"
                    f"Vision Encoder produced {num_vision_tokens} tokens.\n"
                    f"Check Processor logic regarding 'tokens_per_tile' calculation."
                )

            vision_token_embeddings_flat = vision_token_embeddings_flat.to(dtype=token_embeddings.dtype)
            token_embeddings[image_token_positions] = vision_token_embeddings_flat

            # Use custom embeddings instead of input_ids when we've processed vision
            inputs_embeds = token_embeddings
            input_ids = None

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )


__all__ = [
    "SiQ_VLForCausalLM",
    "SiQ_VLPreTrainedModel",
    "SiQ_VLProjector",
    "SiQ_VLTextModel",
    "SiQ_VLVisionModel",
]

AutoModelForCausalLM.register(config_class=SiQ_VLConfig, model_class=SiQ_VLForCausalLM)


def get_stage1_model_and_processor(
    pretrained_vision_model_path: str = "google/siglip2-base-patch16-224",
    pretrained_text_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> tuple[SiQ_VLForCausalLM, SiQ_VLProcessor]:
    """
    Get the initialized SiQ-VL model for stage 1 (multimodality projector allignment) pre-training
    Args:
        pretrained_vision_model_path: Path to the pretrained vision model.
        pretrained_text_model_path: Path to the pretrained text model.

    Returns:
        SiQ_VLForCausalLM instance and SiQ_VLProcessor instance.
    """
    rank_zero_info("Initializing SiQ-VL model for stage 1...")
    config = get_siq_vl_config(
        text_model_name_or_path=pretrained_text_model_path,
        vision_model_name_or_path=pretrained_vision_model_path,
    )
    rank_zero_info(f"Config: \n{config}")

    model = SiQ_VLForCausalLM(config)
    model.text_model = SiQ_VLTextModel.from_pretrained(pretrained_text_model_path)
    model.vision_model = SiQ_VLVisionModel.from_pretrained(pretrained_vision_model_path)

    model.freez_vision_model()
    model.freez_text_model()

    model.print_trainable_parameters()

    processor = SiQ_VLProcessor(
        tokenizer=Qwen2TokenizerFast.from_pretrained(pretrained_text_model_path),
        vit_image_size=model.vision_model.config.image_size,
        vit_patch_size=model.vision_model.config.patch_size,
        pixel_shuffle_factor=model.projector.config.vision_pixel_shuffle_factor,
    )

    return model, processor


def get_stage2_model_and_processor(
    stage_1_checkpoint_path: str = "outputs/checkpoints/siq_vl_stage1",
    use_lora: bool = False,
    lora_r: int = 64,  # Rank of the update matrices
    lora_alpha: int = 16,  # Alpha parameter for LoRA scaling
    lora_dropout: float = 0.05,  # Dropout probability for the LoRA update matrices
    lora_target_modules: list[str] | None = None,
) -> tuple[SiQ_VLForCausalLM, SiQ_VLProcessor]:
    """
    Get the stage 2 SiQ-VL model and processor.
    Args:
        stage_1_checkpoint_path: Path to the stage 1 checkpoint.
        use_lora: Whether to use LoRA to train the model.
        lora_r: Rank of the LoRA update matrices.
        lora_alpha: Alpha parameter for LoRA scaling.
        lora_dropout: Dropout probability for the LoRA update matrices.
        lora_target_modules: Target modules for the LoRA update matrices.
    """
    model = SiQ_VLForCausalLM.from_pretrained(stage_1_checkpoint_path)
    processor = SiQ_VLProcessor.from_pretrained(stage_1_checkpoint_path)

    model.unfreez_text_model()
    model.freez_vision_model()

    if use_lora:
        rank_zero_info("Applying LoRA to the text model...")
        # apply lora to the text model
        from peft import LoraConfig, TaskType, get_peft_model

        if lora_target_modules is None:
            # Target the attention and MLP modules of the transformer layers
            lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        model.text_model = get_peft_model(
            model.text_model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",  # Don't train bias terms for the LoRA update matrices
            ),
        )

    model.print_trainable_parameters()

    rank_zero_info(model.config)
    rank_zero_info(model.text_model.config)
    rank_zero_info(model.vision_model.config)
    rank_zero_info(model.projector.config)

    return model, processor


if __name__ == "__main__":
    print("=" * 100)
    print("Stage 1")
    print("=" * 100)

    stage_1_model, stage_1_processor = get_stage1_model_and_processor()

    inputs = stage_1_processor(
        batch=[
            (Image.open("image.png"), "Describe this image.", "The image shows a beautiful sunset."),
            (Image.open("image.png"), "How many people are in the image?", "There are 2 people in the image."),
        ],
        return_tensors="pt",
    )

    print("pixel_values min and max:", inputs.pixel_values.min(), inputs.pixel_values.max())
    print("pixel_values shape:", inputs.pixel_values.shape)

    outputs = stage_1_model(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        attention_mask=inputs.attention_mask,
        labels=inputs.labels,
    )
    print(outputs)

    print("Generating...")
    stage_1_model.eval()
    with torch.no_grad():
        output_ids = stage_1_model.generate(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            max_new_tokens=64,
        )

    print(stage_1_processor.batch_decode(output_ids, assistant_only=True, skip_special_tokens=True))

    stage_1_model.save_pretrained("outputs/checkpoints/siq_vl_stage1")
    stage_1_processor.save_pretrained("outputs/checkpoints/siq_vl_stage1")
    del stage_1_model, stage_1_processor

    print("Model and processor saved to outputs/checkpoints/siq_vl_stage1")

    print("=" * 100)
    print("Stage 2")
    print("=" * 100)

    stage_2_model, stage_2_processor = get_stage2_model_and_processor(
        stage_1_checkpoint_path="outputs/checkpoints/siq_vl_stage1",
        use_lora=True,
    )

    inputs = stage_2_processor(
        batch=[
            (Image.open("image.png"), "Describe this image.", "The image shows a beautiful sunset."),
            (Image.open("image.png"), "How many people are in the image?", "There are 2 people in the image."),
        ],
        return_tensors="pt",
    )

    print("pixel_values min and max:", inputs.pixel_values.min(), inputs.pixel_values.max())
    print("pixel_values shape:", inputs.pixel_values.shape)

    outputs = stage_2_model(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        attention_mask=inputs.attention_mask,
        labels=inputs.labels,
    )
    print(outputs)

    print("Generating...")
    stage_2_model.eval()
    with torch.no_grad():
        output_ids = stage_2_model.generate(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            max_new_tokens=64,
        )
    print(stage_2_processor.batch_decode(output_ids, assistant_only=True, skip_special_tokens=True))

    stage_2_model.save_pretrained("outputs/checkpoints/siq_vl_stage2")
    stage_2_processor.save_pretrained("outputs/checkpoints/siq_vl_stage2")
    del stage_2_model, stage_2_processor

    print("Model and processor saved to outputs/checkpoints/siq_vl_stage2")
