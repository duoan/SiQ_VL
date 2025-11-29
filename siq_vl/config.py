# coding=utf-8
"""
Configuration class for SiQ-VL model.
"""

from typing import Optional

from transformers import PretrainedConfig


class SiQ_VLConfig(PretrainedConfig):
    """
    Configuration class for SiQ-VL model.
    
    This config stores all the hyperparameters needed to instantiate a SiQ_VLModel.
    """
    
    model_type = "siq_vl"
    
    def __init__(
        self,
        vision_model_path: str = "google/siglip2-so400m-patch14-384",
        llm_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        freeze_llm: bool = True,
        pixel_shuffle_factor: int = 1,
        image_size: int = 384,
        patch_size: int = 14,
        vision_hidden_size: Optional[int] = None,
        llm_hidden_size: Optional[int] = None,
        image_token_id: int = 151655,  # Qwen's <|image_pad|> token
        vision_start_id: int = 151652,  # <|vision_start|>
        vision_end_id: int = 151653,  # <|vision_end|>
        ignore_index: int = -100,
        **kwargs,
    ):
        """
        Initialize SiQ_VLConfig.
        
        Args:
            vision_model_path: Path or identifier for the vision encoder model
            llm_model_path: Path or identifier for the language model
            freeze_llm: Whether to freeze the LLM during training
            pixel_shuffle_factor: Factor for pixel shuffle operation in projector
            image_size: Input image size (default: 384)
            patch_size: Vision model patch size (default: 14)
            vision_hidden_size: Hidden size of vision encoder (auto-detected if None)
            llm_hidden_size: Hidden size of LLM (auto-detected if None)
            image_token_id: Token ID for image placeholder
            vision_start_id: Token ID for vision start marker
            vision_end_id: Token ID for vision end marker
            ignore_index: Index to use for ignored labels in loss calculation
            **kwargs: Additional arguments passed to PretrainedConfig
        """
        super().__init__(**kwargs)
        
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_llm = freeze_llm
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.image_token_id = image_token_id
        self.vision_start_id = vision_start_id
        self.vision_end_id = vision_end_id
        self.ignore_index = ignore_index

