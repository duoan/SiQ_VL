"""
Sample packing for SiQ-VL training.

Packs multiple tokenized samples into fixed-length sequences with a block-diagonal
causal attention mask, eliminating padding waste entirely. Each packed sequence
contains N sub-sequences where the attention mask prevents cross-sample visibility.

Key design decisions:
- Uses 4D attention mask (B, 1, L, L) for block-diagonal causal masking
- Resets position_ids per sub-sequence for correct RoPE encoding
- Vision tokens from all packed samples are concatenated; offsets track which
  tokens map to which positions in the packed sequence
"""

from dataclasses import dataclass
from typing import Any

import torch


def pack_samples(
    samples: list[dict[str, torch.Tensor]],
    max_length: int,
    pad_token_id: int = 0,
    ignore_label_id: int = -100,
) -> dict[str, torch.Tensor]:
    """
    Pack multiple pre-tokenized samples into a single sequence of max_length.

    Each sample dict should contain:
      - input_ids: (seq_len,) LongTensor
      - labels: (seq_len,) LongTensor
      - attention_mask: (seq_len,) LongTensor (1s for real, 0s for pad — but should be unpadded here)

    Returns dict with:
      - input_ids: (max_length,)
      - labels: (max_length,)
      - position_ids: (max_length,) — reset per sub-sequence
      - attention_mask_4d: (1, max_length, max_length) — block-diagonal causal
      - seq_boundaries: list of (start, end) tuples for each packed sub-sequence
    """
    packed_input_ids = []
    packed_labels = []
    packed_position_ids = []
    seq_boundaries = []

    current_pos = 0
    for sample in samples:
        ids = sample["input_ids"]
        labels = sample["labels"]

        # Remove padding from sample (take only non-pad tokens)
        if "attention_mask" in sample:
            mask = sample["attention_mask"].bool()
            ids = ids[mask]
            labels = labels[mask]

        seq_len = ids.shape[0]
        if current_pos + seq_len > max_length:
            break

        packed_input_ids.append(ids)
        packed_labels.append(labels)
        packed_position_ids.append(torch.arange(seq_len, dtype=torch.long))
        seq_boundaries.append((current_pos, current_pos + seq_len))
        current_pos += seq_len

    # Pad remainder to max_length
    remaining = max_length - current_pos
    if remaining > 0:
        packed_input_ids.append(torch.full((remaining,), pad_token_id, dtype=torch.long))
        packed_labels.append(torch.full((remaining,), ignore_label_id, dtype=torch.long))
        packed_position_ids.append(torch.zeros(remaining, dtype=torch.long))

    input_ids = torch.cat(packed_input_ids)
    labels = torch.cat(packed_labels)
    position_ids = torch.cat(packed_position_ids)

    # Build block-diagonal causal attention mask
    # 0 = attend, -inf = mask (additive mask for SDPA)
    attn_mask = torch.full((max_length, max_length), float("-inf"), dtype=torch.bfloat16)
    for start, end in seq_boundaries:
        # Within each sub-sequence: causal mask (lower triangular)
        seq_len = end - start
        causal_block = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16), diagonal=1
        )
        attn_mask[start:end, start:end] = causal_block

    # Padding region: keep as -inf (no attention to/from padding)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "attention_mask_4d": attn_mask.unsqueeze(0),  # (1, L, L)
        "seq_boundaries": seq_boundaries,
    }


@dataclass
class PackingCollator:
    """
    Data collator that packs multiple samples into fixed-length sequences.

    Pipeline:
    1. Receives raw features from VQADataset (image, question, answer)
    2. Tokenizes each sample via the processor
    3. Bin-packs tokenized samples into sequences of max_length
    4. Builds 4D block-diagonal causal attention mask
    5. Concatenates pixel_values from all packed samples

    The model forward receives:
    - input_ids: (B, max_length)
    - labels: (B, max_length)
    - position_ids: (B, max_length)
    - attention_mask: (B, 1, max_length, max_length) — 4D block-diagonal
    - pixel_values: (total_tiles, C, H, W) — all tiles from all packed samples
    - num_image_tokens: (total_images,) — per-image token counts
    """

    processor: Any
    max_length: int = 1024
    pad_token_id: int = 0

    def __call__(self, features: list[dict[str, Any] | None]) -> dict[str, Any]:
        features = [f for f in features if f is not None]
        if len(features) == 0:
            raise ValueError("PackingCollator received empty features list!")

        # Step 1: Tokenize each sample individually (no padding)
        tokenized_samples = []
        all_pixel_values = []
        all_num_image_tokens = []

        for f in features:
            batch = [(f["image"], f["question"], f["answer"])]
            processed = self.processor(
                batch=batch,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            tokenized_samples.append({
                "input_ids": processed["input_ids"].squeeze(0),
                "labels": processed["labels"].squeeze(0) if "labels" in processed else processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "pixel_values": processed["pixel_values"],
                "num_image_tokens": processed["num_image_tokens"],
            })

        # Step 2: Greedy bin-packing into sequences of max_length
        # Sort by length (longest first) for better packing efficiency
        tokenized_samples.sort(key=lambda x: x["input_ids"].shape[0], reverse=True)

        packed_sequences = []
        used = [False] * len(tokenized_samples)

        for i in range(len(tokenized_samples)):
            if used[i]:
                continue

            # Start a new packed sequence with sample i
            current_pack = [tokenized_samples[i]]
            current_len = tokenized_samples[i]["input_ids"].shape[0]
            used[i] = True

            # Try to fit more samples (first-fit decreasing)
            for j in range(i + 1, len(tokenized_samples)):
                if used[j]:
                    continue
                sample_len = tokenized_samples[j]["input_ids"].shape[0]
                if current_len + sample_len <= self.max_length:
                    current_pack.append(tokenized_samples[j])
                    current_len += sample_len
                    used[j] = True

            packed_sequences.append(current_pack)

        # Step 3: Build packed tensors
        batch_input_ids = []
        batch_labels = []
        batch_position_ids = []
        batch_attn_masks = []

        for pack in packed_sequences:
            # Extract just the text tensors for packing
            text_samples = [{"input_ids": s["input_ids"], "labels": s["labels"], "attention_mask": s["attention_mask"]} for s in pack]
            packed = pack_samples(text_samples, self.max_length, self.pad_token_id)

            batch_input_ids.append(packed["input_ids"])
            batch_labels.append(packed["labels"])
            batch_position_ids.append(packed["position_ids"])
            batch_attn_masks.append(packed["attention_mask_4d"])

            # Collect pixel_values and num_image_tokens from all samples in this pack
            for s in pack:
                all_pixel_values.append(s["pixel_values"])
                all_num_image_tokens.append(s["num_image_tokens"])

        # Stack batch dimensions
        result = {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "position_ids": torch.stack(batch_position_ids),
            "attention_mask": torch.stack(batch_attn_masks),  # (B, 1, L, L)
            "pixel_values": torch.cat(all_pixel_values, dim=0),  # (total_tiles, C, H, W)
            "num_image_tokens": torch.cat(all_num_image_tokens, dim=0),
        }

        return result
