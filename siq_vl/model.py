import json
import os
import re
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoModelForCausalLM
from transformers.utils import logging

from siq_vl.processing import SiQ_VLProcessor

logger = logging.get_logger(__name__)


class SiQ_VLModalityProjector(nn.Module):
    def __init__(self, vision_hidden_dim, pixel_shuffle_factor, lm_hidden_dim):
        super().__init__()
        self.scale_factor = pixel_shuffle_factor
        input_dim = vision_hidden_dim * (pixel_shuffle_factor**2)
        self.proj = nn.Linear(input_dim, lm_hidden_dim, bias=False)
        # Add LayerNorm to normalize the projected embeddings
        self.norm = nn.LayerNorm(lm_hidden_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        # Sequence length must be a perfect square for pixel shuffle
        if seq_root**2 != seq:
            raise ValueError(
                f"Sequence length {seq} is not a perfect square (sqrt={seq_root}). "
                f"Cannot apply pixel shuffle. Please check your vision model configuration."
            )
        # Sequence root must be divisible by scale factor
        if seq_root % self.scale_factor != 0:
            # Find valid factors
            valid_factors = [f for f in range(1, seq_root + 1) if seq_root % f == 0]
            raise ValueError(
                f"seq_root {seq_root} is not divisible by scale factor {self.scale_factor}. "
                f"Valid factors for seq_root {seq_root} are: {valid_factors}. "
                f"Please set pixel_shuffle_factor to one of these values."
            )

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor

        x = x.reshape(
            bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)

        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)
        x = self.norm(x)  # Normalize to match text embedding distribution
        return x


class SiQ_VLModel(nn.Module):
    # CRITICAL: Tell Trainer we don't accept loss kwargs
    # This ensures Trainer properly normalizes loss for gradient accumulation
    # See: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L4060-4064
    accepts_loss_kwargs = False
    supports_gradient_checkpointing = True

    @staticmethod
    def _calculate_expected_seq_len(vision_config, vision_tower=None, vision_model_path=None):
        """
        Calculate expected sequence length from vision model configuration.
        For SigLIP models: seq_len = (image_size / patch_size)^2
        
        If config doesn't have the info, try to infer from model name or do a test forward pass.
        """
        # Try to get image_size from config
        image_size = getattr(vision_config, 'image_size', None)
        
        # Get patch_size from config
        patch_size = getattr(vision_config, 'patch_size', None)
        
        # Try to infer from model name if available
        if vision_model_path and (image_size is None or patch_size is None):
            # Common SigLIP patterns: siglip2-base-patch16-224, siglip2-so400m-patch14-384, etc.
            # Extract patch size (e.g., patch14, patch16)
            patch_match = re.search(r'patch(\d+)', vision_model_path.lower())
            if patch_match and patch_size is None:
                patch_size = int(patch_match.group(1))
            
            # Extract image size (usually at the end: -224, -384, -512)
            size_match = re.search(r'-(\d+)$', vision_model_path)
            if size_match and image_size is None:
                image_size = int(size_match.group(1))
        
        # Default patch_size if still None
        if patch_size is None:
            patch_size = 16  # Default for most SigLIP models
        
        if image_size is not None and patch_size is not None:
            seq_len = (image_size // patch_size) ** 2
            return seq_len
        
        # If we can't determine from config or name, do a test forward pass with common sizes
        if vision_tower is not None:
            # Try common image sizes: 224, 384, 512
            for test_size in [224, 384, 512]:
                try:
                    dummy_image = torch.zeros(1, 3, test_size, test_size)
                    with torch.no_grad():
                        output = vision_tower(dummy_image)
                        if hasattr(output, 'last_hidden_state'):
                            seq_len = output.last_hidden_state.shape[1]
                            # Verify it's a perfect square
                            seq_root = int(seq_len ** 0.5)
                            if seq_root ** 2 == seq_len:
                                return seq_len
                except Exception:
                    continue
        
        return None

    @staticmethod
    def _calculate_pixel_shuffle_factor(seq_len):
        """
        Calculate a valid pixel_shuffle_factor for a given sequence length.
        The factor must divide sqrt(seq_len) evenly.
        """
        if seq_len is None:
            return None
        
        seq_root = int(seq_len ** 0.5)
        
        # Check if seq_len is a perfect square
        if seq_root ** 2 != seq_len:
            return None
        
        # Find the largest factor that divides seq_root evenly
        # Try common factors: 2, 4, 8, 3, 6, 7, 14, etc.
        # Start with larger factors for better compression
        for factor in range(seq_root, 0, -1):
            if seq_root % factor == 0:
                return factor
        
        return 1  # Fallback: no shuffling

    def __init__(
        self,
        vision_model_path="google/siglip2-so400m-patch14-384",
        llm_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        freeze_llm=True,
        gradient_accumulation_steps=1,
        pixel_shuffle_factor=1,  # Default to 1 (no shuffling)
    ):
        super().__init__()

        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # ==========================================================
        # 1. Load Vision Tower (SigLIP 2)
        # ==========================================================
        print(f"Loading Vision Tower: {vision_model_path}...")
        full_vision_model = AutoModel.from_pretrained(vision_model_path)
        self.vision_tower = full_vision_model.vision_model
        del full_vision_model

        # We always freeze the vision tower to save memory
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        print(">>> Vision Tower Frozen.")

        # Get vision hidden size (SigLIP SO400M is typically 1152)
        self.vision_hidden_size = self.vision_tower.config.hidden_size

        # Use provided pixel_shuffle_factor (default is 1)
        self.pixel_shuffle_factor = pixel_shuffle_factor
        print(f">>> Using pixel_shuffle_factor: {pixel_shuffle_factor}")

        # ==========================================================
        # 2. Load LLM (Qwen 2.5 0.5B)
        # ==========================================================
        print(f"Loading LLM: {llm_model_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path)
        self.llm_hidden_size = self.llm.config.hidden_size  # Qwen 0.5B is 896

        if freeze_llm:
            self.llm.requires_grad_(False)
            self.llm.eval()
            for param in self.llm.parameters():
                param.requires_grad = False
            print(">>> LLM Frozen.")

        # ==========================================================
        # 3. Projector (The Bridge)
        # ==========================================================
        # Maps Vision Dimension -> LLM Dimension
        self.projector = SiQ_VLModalityProjector(
            self.vision_hidden_size, pixel_shuffle_factor, self.llm_hidden_size
        )

        # Placeholder ID for the <image> token.
        # with a special ID (e.g., -200) that doesn't exist in the tokenizer.
        self.ignore_index = -100  # Standard label for ignoring loss

        # https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json
        # Use Qwen's native <|image_pad|> token ID
        self.image_token_id = 151655

        # Optional: You might want to define start/end tokens too if you use them
        self.vision_start_id = 151652  # <|vision_start|>
        self.vision_end_id = 151653  # <|vision_end|>

        self.projector.apply(self._init_projector_weights)

    def _init_projector_weights(self, m):
        """
        Custom initialization to align the projector's output distribution
        with the LLM's embedding distribution.

        Standard Transformers (BERT, GPT, Qwen) use init std=0.02.
        """
        if isinstance(m, nn.Linear):
            # Initialize weights with mean=0.0 and std=0.02
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            # Initialize bias to zero
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def get_vision_features(self, pixel_values):
        """Extracts features from images using SigLIP."""
        with torch.no_grad():  # Ensure gradients don't flow back into SigLIP
            # Output: last_hidden_state [batch, num_patches, hidden_dim]
            image_features = self.vision_tower(pixel_values).last_hidden_state
        return image_features

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Redirects the gradient checkpointing request to the inner LLM (Qwen).
        The Vision Tower is frozen, so it doesn't need this.
        """
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def _prepare_llm_inputs(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_embeds: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.LongTensor], torch.LongTensor, List[int]]:
        """
        Common helper to splice image embeddings into text embeddings and build
        padded embeddings, labels, and attention mask for the LLM.

        Used by both the training forward pass and the generation path.

        Args:
            input_ids: [batch, seq_len]
            inputs_embeds: [batch, seq_len, hidden]
            image_embeds: [batch, num_patches, hidden]
            labels: Optional[batch, seq_len]

        Returns:
            final_input_embeds: [batch, max_seq_len, hidden]
            final_labels: Optional[batch, max_seq_len]
            final_attention_mask: [batch, max_seq_len]
            seq_lengths: list of original (unpadded) lengths per sample after splicing
        """
        new_input_embeds: List[torch.Tensor] = []
        new_labels: Optional[List[torch.Tensor]] = [] if labels is not None else None

        batch_size = inputs_embeds.shape[0]

        for i in range(batch_size):
            # Check where the image placeholder is located
            cur_input_ids = input_ids[i]

            # If no image placeholder is found (text-only sample), keep as is
            if (cur_input_ids == self.image_token_id).sum() == 0:
                new_input_embeds.append(inputs_embeds[i])
                if labels is not None and new_labels is not None:
                    new_labels.append(labels[i])
                continue

            # 1. Find ALL image token positions
            image_token_mask = cur_input_ids == self.image_token_id
            num_image_tokens = image_token_mask.sum().item()

            if num_image_tokens == 0:
                # No image tokens, shouldn't happen but handle gracefully
                new_input_embeds.append(inputs_embeds[i])
                if labels is not None and new_labels is not None:
                    new_labels.append(labels[i])
                continue

            # Find the first and last image token positions
            image_token_indices = image_token_mask.nonzero(as_tuple=True)[0]
            image_start_idx = image_token_indices[0].item()
            image_end_idx = image_token_indices[-1].item() + 1  # +1 to include the last token

            # 2. Slice the text embeddings: Prefix + Suffix
            # CRITICAL: Skip ALL image tokens, not just the first one!
            prefix_embeds = inputs_embeds[i, :image_start_idx]
            suffix_embeds = inputs_embeds[i, image_end_idx:]

            # 3. Concatenate: [Prefix, Image_Features, Suffix]
            cur_image_embed = image_embeds[i]
            combined_embed = torch.cat(
                [prefix_embeds, cur_image_embed, suffix_embeds], dim=0
            )
            new_input_embeds.append(combined_embed)

            # 4. Handle Labels (Align with new sequence length)
            if labels is not None and new_labels is not None:
                cur_labels = labels[i]
                prefix_labels = cur_labels[:image_start_idx]
                suffix_labels = cur_labels[image_end_idx:]

                # Create labels for the image tokens
                # We use ignore_index (-100) because we don't want the model to predict the image patches
                image_labels = torch.full(
                    (cur_image_embed.shape[0],),
                    self.ignore_index,
                    dtype=labels.dtype,
                    device=labels.device,
                )

                combined_labels = torch.cat(
                    [prefix_labels, image_labels, suffix_labels], dim=0
                )
                new_labels.append(combined_labels)

        # Re-pack the batch: pad embeddings to the longest sequence in the batch
        final_input_embeds = pad_sequence(new_input_embeds, batch_first=True)

        # Pad labels if provided
        final_labels: Optional[torch.LongTensor] = None
        if labels is not None and new_labels is not None:
            final_labels = pad_sequence(
                new_labels, batch_first=True, padding_value=self.ignore_index
            )

        # Generate attention mask (1 for valid tokens, 0 for padding)
        final_attention_mask = torch.ones(
            final_input_embeds.shape[:2],
            dtype=torch.long,
            device=final_input_embeds.device,
        )

        # Mask out the padding areas (assuming right-padding by pad_sequence)
        seq_lengths: List[int] = []
        for i, embed in enumerate(new_input_embeds):
            length = len(embed)
            seq_lengths.append(length)
            final_attention_mask[i, length:] = 0

        return final_input_embeds, final_labels, final_attention_mask, seq_lengths

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        # Optional, usually re-calculated
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        input_ids: [batch, seq_len] (contains text IDs and the special image placeholder ID)
        pixel_values: [batch, channels, height, width]
        labels: [batch, seq_len] (for loss calculation)
        """

        # 1. Get Image Embeddings and Project them
        # image_features shape: [batch, num_patches (e.g., 729), 1152]
        image_features = self.get_vision_features(pixel_values)

        # image_embeds shape: [batch, num_patches, 896] -> Now compatible with LLM
        image_embeds = self.projector(image_features)

        # 2. Get Text Embeddings
        # inputs_embeds shape: [batch, seq_len, 896]
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # --- DIAGNOSTIC PRINT ---
        # if self.training:
        #     # Count image tokens in input_ids
        #     num_image_tokens = (input_ids[0] == self.image_token_id).sum().item()
        #     num_image_embeds = image_embeds.shape[1]

        #     logger.warning_once(  # type: ignore
        #         f"\n[DEBUG Forward Pass]"
        #         f"\n  Input IDs shape: {input_ids.shape}"
        #         f"\n  Num <image_pad> tokens in input_ids: {num_image_tokens}"
        #         f"\n  Image embeds shape: {image_embeds.shape} (num_patches={num_image_embeds})"
        #         f"\n  Text embeds shape: {inputs_embeds.shape}"
        #         f"\n  Image Embeds: Mean={image_embeds.mean().item():.4f}, Std={image_embeds.std().item():.4f}"
        #         f"\n  Text  Embeds: Mean={inputs_embeds.mean().item():.4f}, Std={inputs_embeds.std().item():.4f}"
        #         f"\n  Labels shape: {labels.shape if labels is not None else 'None'}"
        #         f"\n  Num valid labels: {(labels != -100).sum().item() if labels is not None else 0}"
        #     )
        # ----------------------------------------------

        # ==========================================================
        # 3. The Surgery: Embedding Fusion (Splicing)
        # ==========================================================
        # Use the helper method to splice image embeddings into text embeddings
        # and build padded embeddings, labels, and attention mask for the LLM
        (
            final_input_embeds,
            final_labels,
            final_attention_mask,
            _seq_lengths,  # Not needed in forward pass, but returned by helper
        ) = self._prepare_llm_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            labels=labels,
        )

        # ==========================================================
        # 4. Forward Pass through LLM
        # ==========================================================

        # --- DEBUG: Check final shapes before LLM ---
        # if self.training:
        #     logger.warning_once(  # type: ignore
        #         f"\n[DEBUG Before LLM]"
        #         f"\n  Final input_embeds shape: {final_input_embeds.shape}"
        #         f"\n  Final attention_mask shape: {final_attention_mask.shape}"
        #         f"\n  Final labels shape: {final_labels.shape if final_labels is not None else 'None'}"
        #         f"\n  Num valid labels in final: {(final_labels != -100).sum().item() if final_labels is not None else 0}"
        #         f"\n  Final embeds stats: Mean={final_input_embeds.mean().item():.4f}, Std={final_input_embeds.std().item():.4f}"
        #     )

        outputs = self.llm(
            inputs_embeds=final_input_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            return_dict=True,
        )

        # NOTE: We set `accepts_loss_kwargs = False` as a class attribute
        # This tells Trainer to properly normalize loss for gradient accumulation
        # No manual normalization needed here - Trainer handles it automatically

        # --- DEBUG: Check loss and accuracy ---
        # if self.training and hasattr(outputs, "loss") and final_labels is not None:
        #     # Calculate prediction accuracy on valid labels
        #     predictions = outputs.logits.argmax(dim=-1)
        #     valid_mask = final_labels != -100

        #     correct = 0
        #     total = 0
        #     accuracy = 0.0

        #     if valid_mask.sum() > 0:
        #         correct = (
        #             (predictions[valid_mask] == final_labels[valid_mask]).sum().item()
        #         )
        #         total = valid_mask.sum().item()
        #         accuracy = correct / total * 100

        #     logger.warning_once(  # type: ignore
        #         f"\n[DEBUG LLM Output]"
        #         f"\n  Loss (from LLM): {outputs.loss.item():.4f}"
        #         f"\n  Loss requires_grad: {outputs.loss.requires_grad}"
        #         f"\n  Loss shape: {outputs.loss.shape}"
        #         f"\n  Accuracy: {accuracy:.2f}% ({correct}/{total} correct)"
        #         f"\n  Logits shape: {outputs.logits.shape}"
        #         f"\n  Logits stats: Mean={outputs.logits.mean().item():.4f}, Std={outputs.logits.std().item():.4f}, Max={outputs.logits.max().item():.4f}"
        #     )

        # CRITICAL: Check if loss needs to be normalized
        # In DDP with gradient accumulation, we might need to scale the loss
        # if self.training and hasattr(outputs, "loss"):
        #     # Print the loss that will be returned to Trainer
        #     logger.warning_once(  # type: ignore
        #         f"\n[DEBUG] Returning loss to Trainer: {outputs.loss.item():.4f}"
        #     )

        return outputs

    def save_pretrained(self, save_directory, **kwargs):
        """
        Custom save method to handle:
        1. Saving the full model state_dict (including Projector).
        2. Avoiding the 'safetensors' shared weight error by using torch.save (.bin).
        3. Saving the base configuration.
        """
        print(f">>> Custom Saving to {save_directory}...")

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 1. Save the LLM Config
        # This creates 'config.json' so Hugging Face knows the base parameters
        self.llm.config.save_pretrained(save_directory)

        # 2. Save the Full State Dict as a standard PyTorch binary
        # We explicitly use 'pytorch_model.bin' to avoid Safetensors issues
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # 3. Save a custom config file (Optional but recommended)
        # This helps you remember what vision encoder you used
        custom_config = {
            "vision_model_path": self.vision_model_path,
            "llm_model_path": self.llm_model_path,
            "model_type": "siq_vl",
            "image_token_id": self.image_token_id,
        }

        with open(os.path.join(save_directory, "model_config.json"), "w") as f:
            json.dump(custom_config, f, indent=2)

        print(f">>> Model saved successfully to {model_path}")

    def generate_answer(
        self,
        processor: SiQ_VLProcessor,
        samples: Union[
            Tuple[Union[Image.Image, str], str],
            List[Tuple[Union[Image.Image, str], str]],
        ],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        num_beams: int = 2,
        device: Optional[torch.device] = None,
    ) -> Union[str, List[str]]:
        """
        Generate answers for one or more (image, question) pairs.

        Args:
            processor: The SiQ_VLProcessor instance.
            samples: Either a single (image, question) tuple or
                     a list of (image, question) tuples, where:
                     - image: PIL Image or path to image file
                     - question: question string
            max_new_tokens: Maximum number of tokens to generate (default: 256).
            temperature: Sampling temperature (default: 0.7).
            do_sample: Whether to use sampling (default: True).
            device: Device to run on. If None, uses model's device.

        Returns:
            Single generated answer (str) if inputs are single-sample,
            otherwise a list of answers (List[str]) for batched inputs.
        """
        # Normalize to batch of (image, question) tuples
        if isinstance(samples, tuple):
            batch_samples: List[Tuple[Union[Image.Image, str], str]] = [samples]
            single_sample = True
        elif isinstance(samples, list):
            if len(samples) == 0:
                raise ValueError("samples list must not be empty.")
            # Basic validation: each item is a 2-tuple (image, question)
            for s in samples:
                if not (isinstance(s, tuple) and len(s) == 2):
                    raise ValueError(
                        "Each element in samples must be a (image, question) tuple."
                    )
            batch_samples = samples
            single_sample = False
        else:
            raise ValueError(
                "samples must be either a (image, question) tuple or a list of such tuples."
            )

        batch_size = len(batch_samples)

        # Get device
        if device is None:
            device = next(self.parameters()).device

        # Convert images to PIL if needed and collect questions
        processed_images: List[Image.Image] = []
        questions: List[str] = []
        for img, q in batch_samples:
            if isinstance(img, str):
                processed_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                processed_images.append(img)
            else:
                from siq_vl.dataset import _to_pil_rgb

                processed_images.append(_to_pil_rgb(img))
            questions.append(q)

        self.eval()

        # Prepare messages in ChatML format (batched)
        messages: List[dict] = []
        for img, q in zip(processed_images, questions):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": q},
                    ],
                }
            )

        # Process inputs (batched)
        inputs = processor(
            text=messages,
            images=processed_images,
            return_tensors="pt",
            add_generation_prompt=True,
            padding="longest",
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            # 1. Get image embeddings and project them (same as model forward)
            image_features = self.get_vision_features(pixel_values)
            image_embeds = self.projector(image_features)

            # 2. Get text embeddings
            text_embeds = self.llm.get_input_embeddings()(input_ids)

            # 3. Fuse image embeddings into text embeddings per batch element
            (
                final_input_embeds,
                _final_labels,
                final_attention_mask,
                seq_lengths,
            ) = self._prepare_llm_inputs(
                input_ids=input_ids,
                inputs_embeds=text_embeds,
                image_embeds=image_embeds,
                labels=None,
            )

            # 4. Generate using LLM with the combined embeddings
            generated_ids = self.llm.generate(
                inputs_embeds=final_input_embeds,
                attention_mask=final_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=num_beams,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

            # 5. Extract only the newly generated tokens per sample
            answers: List[str] = []
            for i in range(batch_size):
                prompt_length = seq_lengths[i]
                gen_tokens = generated_ids[i][prompt_length:]

                if len(gen_tokens) > 0:
                    text = processor.decode(
                        gen_tokens, skip_special_tokens=True
                    ).strip()
                else:
                    text = "[No generation]"

                answers.append(text)

        # Return single string if called with single-sample inputs, otherwise list
        if single_sample:
            return answers[0]
        return answers


def load_model_from_checkpoint(
    checkpoint_dir: str,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[SiQ_VLModel, SiQ_VLProcessor]:
    """
    Load model and processor from a checkpoint directory.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        device: Device to load model on (default: "cuda" if available, else "cpu")
    
    Returns:
        Tuple of (model, processor)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)
    
    # Load checkpoint configuration
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_config = json.load(f)
        vision_model_path = model_config.get("vision_model_path")
        llm_model_path = model_config.get("llm_model_path")
        freeze_llm = model_config.get("freeze_llm", True)
    else:
        # Fallback: try to infer from checkpoint directory structure or use defaults
        print(f">>> Warning: model_config.json not found in {checkpoint_dir}")
        print(">>> Using default model paths. This may not match your checkpoint.")
        vision_model_path = "google/siglip-so400m-patch14-384"
        llm_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        freeze_llm = True
    
    # Load processor (saved with the model)
    print(f">>> Loading processor from {checkpoint_dir}...")
    processor = SiQ_VLProcessor.from_pretrained(checkpoint_dir)
    
    # Initialize model with saved configuration
    print(f">>> Loading model: vision={vision_model_path}, llm={llm_model_path}...")
    model = SiQ_VLModel(
        vision_model_path=vision_model_path,
        llm_model_path=llm_model_path,
        freeze_llm=freeze_llm,
    )
    
    # Load the trained weights
    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(model_path):
        print(f">>> Loading weights from {model_path}...")
        model.load_state_dict(
            torch.load(model_path, map_location=device),
            strict=False,  # Allow missing keys (e.g., if some layers weren't trained)
        )
    else:
        print(f">>> Warning: pytorch_model.bin not found in {checkpoint_dir}")
        print(">>> Using model with pretrained weights only.")
    
    model.to(device)
    model.eval()
    
    print(f">>> Model loaded successfully on {device}")
    
    return model, processor
