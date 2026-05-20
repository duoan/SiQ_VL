from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SiQ_VLDataCollator:
    processor: Any
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    return_raw_data: bool = False

    def __call__(self, features: list[dict[str, Any] | None]) -> dict[str, Any]:
        # Filter out None
        features = [f for f in features if f is not None]

        if len(features) == 0:
            raise ValueError("Collator received empty features list!")

        # Extract raw data from features
        images = [f["image"] for f in features]
        questions = [f["question"] for f in features]
        answers = [f["answer"] for f in features]

        # Build batch for processor: list of (image, question, answer) tuples
        batch = list(zip(images, questions, answers, strict=False))

        processed = self.processor(
            batch=batch,
            return_tensors="pt",
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Processor returns BatchEncoding with input_ids, pixel_values, labels, attention_mask
        result = dict(processed)

        # Add raw metadata if needed
        if self.return_raw_data:
            result["questions"] = questions
            result["answers"] = answers
            result["images"] = images

        return result


@dataclass
class PackingCollator:
    """Packs multiple variable-length sequences into fixed-length bins for efficient training.

    Uses HuggingFace transformers' native packing detection via position_ids resets.
    Requires the model to use attn_implementation='flex_attention' for proper
    document-boundary attention masking.

    The collator:
    1. Tokenizes each sample individually (no padding)
    2. Bin-packs samples into target_length groups using first-fit-decreasing
    3. Concatenates tokens, labels, and builds position_ids with resets
    4. Returns NO attention_mask (triggers packed sequence detection in transformers)
    """

    processor: Any
    pack_max_length: int = 2048
    max_length: int | None = None
    drop_last_bin: bool = False

    def __call__(self, features: list[dict[str, Any] | None]) -> dict[str, Any]:
        features = [f for f in features if f is not None]
        if not features:
            raise ValueError("Collator received empty features list!")

        # Step 1: Process each sample individually to get per-sample tokens
        per_sample = []
        for f in features:
            processed = self.processor(
                batch=[(f["image"], f["question"], f["answer"])],
                return_tensors="pt",
                truncation=self.max_length is not None,
                max_length=self.max_length,
                padding=False,
            )
            input_ids = processed["input_ids"].squeeze(0)  # (seq_len,)
            labels = processed["labels"].squeeze(0) if "labels" in processed else torch.full_like(input_ids, -100)
            pixel_values = processed["pixel_values"]  # (n_tiles, C, H, W)
            num_image_tokens = processed["num_image_tokens"]  # (1,) or scalar
            per_sample.append({
                "input_ids": input_ids,
                "labels": labels,
                "pixel_values": pixel_values,
                "num_image_tokens": num_image_tokens,
                "length": input_ids.shape[0],
            })

        # Step 2: First-fit-decreasing bin packing
        per_sample.sort(key=lambda x: x["length"], reverse=True)
        bins: list[list[int]] = []  # each bin is a list of sample indices
        bin_lengths: list[int] = []

        for idx, sample in enumerate(per_sample):
            placed = False
            for b_idx, b_len in enumerate(bin_lengths):
                if b_len + sample["length"] <= self.pack_max_length:
                    bins[b_idx].append(idx)
                    bin_lengths[b_idx] += sample["length"]
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_lengths.append(sample["length"])

        if self.drop_last_bin and len(bins) > 1 and bin_lengths[-1] < self.pack_max_length * 0.5:
            bins = bins[:-1]

        # Step 3: Build packed batch tensors
        # Always pad to pack_max_length for stable shapes (needed for torch.compile)
        max_packed_len = self.pack_max_length
        batch_input_ids = []
        batch_labels = []
        batch_position_ids = []
        all_pixel_values = []
        all_num_image_tokens = []

        for b_idx, bin_samples in enumerate(bins):
            packed_ids = []
            packed_labels = []
            packed_positions = []

            for sample_idx in bin_samples:
                s = per_sample[sample_idx]
                seq_len = s["length"]
                packed_ids.append(s["input_ids"])
                packed_labels.append(s["labels"])
                packed_positions.append(torch.arange(seq_len, dtype=torch.long))
                all_pixel_values.append(s["pixel_values"])
                all_num_image_tokens.append(s["num_image_tokens"])

            # Concatenate sequences in this bin
            bin_ids = torch.cat(packed_ids)
            bin_labels = torch.cat(packed_labels)
            bin_positions = torch.cat(packed_positions)

            # Pad to max_packed_len for batching
            pad_len = max_packed_len - bin_ids.shape[0]
            if pad_len > 0:
                pad_token_id = self.processor.tokenizer.pad_token_id or 0
                bin_ids = torch.cat([bin_ids, torch.full((pad_len,), pad_token_id, dtype=bin_ids.dtype)])
                bin_labels = torch.cat([bin_labels, torch.full((pad_len,), -100, dtype=bin_labels.dtype)])
                bin_positions = torch.cat([bin_positions, torch.zeros(pad_len, dtype=bin_positions.dtype)])

            batch_input_ids.append(bin_ids)
            batch_labels.append(bin_labels)
            batch_position_ids.append(bin_positions)

        result = {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "position_ids": torch.stack(batch_position_ids),
            "pixel_values": torch.cat(all_pixel_values, dim=0),
            "num_image_tokens": torch.cat(all_num_image_tokens, dim=0),
        }
        return result


@dataclass
class CachedVisionDataCollator:
    """
    Data collator for pre-cached vision features.
    Uses processor.process_cached() to skip image processing entirely.
    """

    processor: Any
    max_length: int | None = None

    def __call__(self, features: list[dict[str, Any] | None]) -> dict[str, Any]:
        features = [f for f in features if f is not None]

        if len(features) == 0:
            raise ValueError("Collator received empty features list!")

        questions = [f["question"] for f in features]
        answers = [f["answer"] for f in features]
        vision_features = [f["vision_features"] for f in features]
        num_tiles = [f["num_tiles"] for f in features]

        processed = self.processor.process_cached(
            questions=questions,
            answers=answers,
            num_tiles_per_image=num_tiles,
            vision_features=vision_features,
            return_tensors="pt",
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding="longest",
        )

        return dict(processed)
