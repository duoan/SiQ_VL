"""
Configuration class for SiQ-VL model.
"""

from typing import Any, ClassVar

from transformers import AutoConfig, PretrainedConfig, Qwen2Config, SiglipVisionConfig


class SiQ_VLVisionConfig(SiglipVisionConfig):
    """
    Configuration class for SiQ-VL vision model.
    """

    model_type = "siq_vl"
    base_config_key = "vision_config"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SiQ_VLTextConfig(Qwen2Config):
    """
    Configuration class for SiQ-VL text model.
    """

    model_type = "siq_vl"
    base_config_key = "text_config"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SiQ_VLProjectorConfig(PretrainedConfig):
    """
    Configuration class for SiQ-VL projector.
    """

    model_type = "siq_vl"
    base_config_key = "projector_config"

    def __init__(
        self,
        vision_pixel_shuffle_factor: int = 2,
        vision_hidden_size: int = 768,
        text_hidden_size: int = 896,
        intermediate_size: int = 1024,
        hidden_act: str = "gelu",
        *args,
        **kwargs,
    ):
        """
        Initialize SiQ_VLProjectorConfig.
        """

        super().__init__(*args, **kwargs)

        self.vision_pixel_shuffle_factor = vision_pixel_shuffle_factor
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act


class SiQ_VLConfig(PretrainedConfig):
    """
    Configuration class for SiQ-VL model.

    This config stores all the hyperparameters needed to instantiate a SiQ_VLModel.
    """

    model_type = "siq_vl"
    sub_configs: ClassVar[dict[str, type[PretrainedConfig]]] = {
        "text_config": SiQ_VLTextConfig,
        "vision_config": SiQ_VLVisionConfig,
        "projector_config": SiQ_VLProjectorConfig,
    }

    def __init__(
        self,
        vision_config: SiQ_VLVisionConfig | dict[str, Any] | None = None,
        text_config: SiQ_VLTextConfig | dict[str, Any] | None = None,
        projector_config: SiQ_VLProjectorConfig | dict[str, Any] | None = None,
        # https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json
        ignore_token_index: int = -100,
        image_token_index: int = 151655,  # <|image_pad|>
        vision_start_token_index: int = 151652,  # <|vision_start|>
        vision_end_token_index: int = 151653,  # <|vision_end|>
        *args,
        **kwargs,
    ):
        """
        Initialize SiQ_VLConfig.

        Args:
            vision_config: Configuration for the vision encoder (can be a dict or config object)
            text_config: Configuration for the language model (can be a dict or config object)
            projector_config: Configuration for the projector (can be a dict or config object)
            *args: Additional arguments passed to Qwen2Config
            **kwargs: Additional arguments passed to PretrainedConfig
        """
        # Convert dict sub-configs to config objects BEFORE calling super().__init__()
        # This is critical because super().__init__() may trigger logging which calls __repr__
        # and __repr__ needs the sub-configs to already be config objects
        if isinstance(vision_config, dict):
            vision_config = SiQ_VLVisionConfig(**vision_config)
        if isinstance(text_config, dict):
            text_config = SiQ_VLTextConfig(**text_config)
        if isinstance(projector_config, dict):
            projector_config = SiQ_VLProjectorConfig(**projector_config)

        super().__init__(*args, **kwargs)

        self.vision_config = vision_config
        self.text_config = text_config
        self.projector_config = projector_config
        self.ignore_token_index = ignore_token_index
        self.image_token_index = image_token_index
        self.vision_start_token_index = vision_start_token_index
        self.vision_end_token_index = vision_end_token_index

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "SiQ_VLConfig":
        """
        Instantiates a SiQ_VLConfig from a Python dictionary of parameters.
        Properly handles sub-configs that may be passed as dictionaries.
        """
        config_dict = config_dict.copy() if config_dict else {}

        # Convert sub-config dicts to config objects before instantiation
        # This ensures they're config objects when __init__ is called
        if "vision_config" in config_dict and isinstance(config_dict["vision_config"], dict):
            config_dict["vision_config"] = SiQ_VLVisionConfig(**config_dict["vision_config"])
        if "text_config" in config_dict and isinstance(config_dict["text_config"], dict):
            config_dict["text_config"] = SiQ_VLTextConfig(**config_dict["text_config"])
        if "projector_config" in config_dict and isinstance(config_dict["projector_config"], dict):
            config_dict["projector_config"] = SiQ_VLProjectorConfig(**config_dict["projector_config"])

        # Also check kwargs for sub-configs that need conversion
        if "vision_config" in kwargs and isinstance(kwargs["vision_config"], dict):
            kwargs["vision_config"] = SiQ_VLVisionConfig(**kwargs["vision_config"])
        if "text_config" in kwargs and isinstance(kwargs["text_config"], dict):
            kwargs["text_config"] = SiQ_VLTextConfig(**kwargs["text_config"])
        if "projector_config" in kwargs and isinstance(kwargs["projector_config"], dict):
            kwargs["projector_config"] = SiQ_VLProjectorConfig(**kwargs["projector_config"])

        # Call parent's from_dict which will handle the rest
        return super().from_dict(config_dict, **kwargs)


__all__ = ["SiQ_VLConfig", "SiQ_VLProjectorConfig", "SiQ_VLTextConfig", "SiQ_VLVisionConfig"]
AutoConfig.register("siq_vl", SiQ_VLConfig)


def get_siq_vl_config(
    text_model_name_or_path: str,
    vision_model_name_or_path: str,
    vision_pixel_shuffle_factor: int = 2,
) -> SiQ_VLConfig:
    """
    Get SiQ-VL configuration from text and vision model paths.

    Args:
        text_model_name_or_path: Path or identifier for the text model
        vision_model_name_or_path: Path or identifier for the vision model
        vision_pixel_shuffle_factor: Pixel shuffle factor for the projector

    Returns:
        SiQ_VLConfig: Configuration for the SiQ-VL model
    """

    text_config = SiQ_VLTextConfig.from_pretrained(text_model_name_or_path)
    vision_config = SiQ_VLVisionConfig.from_pretrained(vision_model_name_or_path)
    projector_config = SiQ_VLProjectorConfig(
        vision_hidden_size=vision_config.hidden_size,
        text_hidden_size=text_config.hidden_size,
        vision_pixel_shuffle_factor=vision_pixel_shuffle_factor,
    )

    return SiQ_VLConfig(text_config=text_config, vision_config=vision_config, projector_config=projector_config)


if __name__ == "__main__":
    config = get_siq_vl_config("Qwen/Qwen2.5-0.5B-Instruct", "google/siglip2-so400m-patch16-512")
    print(config)
