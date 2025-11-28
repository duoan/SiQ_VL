import json
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoModelForCausalLM
from transformers.utils import logging

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
        assert seq_root**2 == seq
        # Sequence root must be divisible by scale factor
        assert seq_root % self.scale_factor == 0

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

    def __init__(
        self,
        vision_model_path="google/siglip2-so400m-patch14-384",
        llm_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        freeze_llm=True,
        gradient_accumulation_steps=1,
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
            self.vision_hidden_size, 3, self.llm_hidden_size
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
        # We need to find the <image> placeholder in 'input_ids'
        # and replace its embedding with the actual 'image_embeds'.

        new_input_embeds = []
        new_labels = [] if labels is not None else None

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

            # --- Splicing Logic ---
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
            image_end_idx = (
                image_token_indices[-1].item() + 1
            )  # +1 to include the last token

            # 2. Slice the text embeddings: Prefix + Suffix
            # CRITICAL: Skip ALL image tokens, not just the first one!
            prefix_embeds = inputs_embeds[i, :image_start_idx]
            suffix_embeds = inputs_embeds[i, image_end_idx:]  # Skip all 81 tokens

            # 3. Concatenate: [Prefix, Image_Features, Suffix]
            cur_image_embed = image_embeds[i]  # Shape: [81, hidden]
            combined_embed = torch.cat(
                [prefix_embeds, cur_image_embed, suffix_embeds], dim=0
            )
            new_input_embeds.append(combined_embed)

            # 4. Handle Labels (Align with new sequence length)
            if labels is not None and new_labels is not None:
                cur_labels = labels[i]
                prefix_labels = cur_labels[:image_start_idx]
                suffix_labels = cur_labels[image_end_idx:]  # Skip all 81 tokens

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

        # 4. Re-Pack the Batch
        # Since inserting images changes the sequence length, we must re-pad the batch

        # Pad Embeddings to the longest sequence in the batch
        final_input_embeds = pad_sequence(new_input_embeds, batch_first=True)

        # Pad Labels
        final_labels = None
        if labels is not None and new_labels is not None:
            final_labels = pad_sequence(
                new_labels, batch_first=True, padding_value=self.ignore_index
            )

        # Generate Attention Mask
        # (1 for valid tokens, 0 for padding)
        final_attention_mask = torch.ones(
            final_input_embeds.shape[:2],
            dtype=torch.long,
            device=final_input_embeds.device,
        )

        # Mask out the padding areas (assuming right-padding by pad_sequence)
        for i, embed in enumerate(new_input_embeds):
            final_attention_mask[i, len(embed) :] = 0

        # ==========================================================
        # 5. Forward Pass through LLM
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
