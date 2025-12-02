"""
Configuration class for SiQ-VL model.
"""

from transformers import AutoConfig, Qwen2Config, SiglipVisionConfig


class SiQ_VLConfig(Qwen2Config):
    """
    Configuration class for SiQ-VL model.

    This config stores all the hyperparameters needed to instantiate a SiQ_VLModel.
    """

    model_type = "siq_vl"

    def __init__(
        self,
        pretrained_vision_model_path: str = "google/siglip2-so400m-patch14-384",
        pretrained_language_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        # Model freezing hyperparameters
        freeze_language_model: bool = True,
        freeze_vision_encoder: bool = True,
        # Vision model hyperparameters
        vision_pixel_shuffle_factor: int = 1,
        # Language model hyperparameters
        language_model_image_token_id: int = 151655,  # Qwen's <|image_pad|> token
        language_model_vision_start_id: int = 151652,  # <|vision_start|>
        language_model_vision_end_id: int = 151653,  # <|vision_end|>
        language_model_ignore_index: int = -100,
        # LoRA hyperparameters
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_scaling: float = 1.0,
        lora_target_modules: list[str] | None = None,
        lora_task_type: str = "CAUSAL_LM",
        **kwargs,
    ):
        """
        Initialize SiQ_VLConfig.

        Args:
            pretrained_vision_model_path: Path or identifier for the vision encoder model
            pretrained_language_model_path: Path or identifier for the language model
            freeze_language_model: Whether to freeze the language model during training
            freeze_vision_model: Whether to freeze the vision model during training
            vision_pixel_shuffle_factor: Factor for pixel shuffle operation in projector
            language_model_image_token_id: Token ID for image placeholder
            language_model_vision_start_id: Token ID for vision start marker
            language_model_vision_end_id: Token ID for vision end marker
            language_model_ignore_index: Index to use for ignored labels in loss calculation
            using_lora: Whether to use LoRA
            lora_rank: Rank of LoRA
            lora_alpha: Alpha of LoRA
            lora_dropout: Dropout of LoRA
            lora_scaling: Scaling of LoRA
            lora_target_modules: Target modules of LoRA
            lora_task_type: Task type of LoRA
            **kwargs: Additional arguments passed to PretrainedConfig
        """
        language_model_config = Qwen2Config.from_pretrained(pretrained_language_model_path, trust_remote_code=True)
        vision_model_config = SiglipVisionConfig.from_pretrained(pretrained_vision_model_path, trust_remote_code=True)

        super().__init__(**language_model_config.to_dict(), **kwargs)

        self.pretrained_vision_model_path = pretrained_vision_model_path
        self.pretrained_language_model_path = pretrained_language_model_path

        self.freeze_language_model = freeze_language_model
        self.freeze_vision_encoder = freeze_vision_encoder

        self.vision_pixel_shuffle_factor = vision_pixel_shuffle_factor
        self.vision_image_size = vision_model_config.image_size
        self.vision_patch_size = vision_model_config.patch_size
        self.vision_hidden_size = vision_model_config.hidden_size

        self.language_model_hidden_size = language_model_config.hidden_size
        self.language_model_image_token_id = language_model_image_token_id
        self.language_model_vision_start_id = language_model_vision_start_id
        self.language_model_vision_end_id = language_model_vision_end_id
        self.language_model_ignore_index = language_model_ignore_index

        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_scaling = lora_scaling
        self.lora_target_modules = lora_target_modules
        self.lora_task_type = lora_task_type


__all__ = ["SiQ_VLConfig"]

AutoConfig.register("siq_vl", SiQ_VLConfig)
