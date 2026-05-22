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

        padding = "max_length" if self.max_length else "longest"
        processed = self.processor(
            batch=batch,
            return_tensors="pt",
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding=padding,
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
    """Fast bin-packing collator for pre-processed samples.

    Expects samples from ProcessedVQADataset (already tokenized + image processed).
    Only does lightweight tensor operations: sort, concat, pad.
    No image processing or tokenization happens here.
    """

    pack_max_length: int = 1024
    pad_token_id: int = 0

    def __call__(self, features: list[dict[str, Any] | None]) -> dict[str, Any]:
        features = [f for f in features if f is not None]
        if not features:
            raise ValueError("Collator received empty features list!")

        # Sort by length descending for better bin packing
        features.sort(key=lambda x: x["length"], reverse=True)

        # First-fit-decreasing bin packing
        bins: list[list[int]] = []
        bin_lengths: list[int] = []

        for idx, sample in enumerate(features):
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

        # Build packed batch tensors
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        all_pixel_values = []
        all_num_image_tokens = []

        for bin_samples in bins:
            packed_ids = []
            packed_labels = []

            for sample_idx in bin_samples:
                s = features[sample_idx]
                packed_ids.append(s["input_ids"])
                packed_labels.append(s["labels"])
                all_pixel_values.append(s["pixel_values"])
                all_num_image_tokens.append(s["num_image_tokens"])

            bin_ids = torch.cat(packed_ids)
            bin_labels = torch.cat(packed_labels)

            # Truncate if overflow
            if bin_ids.shape[0] > self.pack_max_length:
                bin_ids = bin_ids[:self.pack_max_length]
                bin_labels = bin_labels[:self.pack_max_length]

            # Pad to pack_max_length for stable tensor shapes
            pad_len = self.pack_max_length - bin_ids.shape[0]
            attention_mask = torch.ones(self.pack_max_length, dtype=torch.long)
            if pad_len > 0:
                bin_ids = torch.cat([bin_ids, torch.full((pad_len,), self.pad_token_id, dtype=bin_ids.dtype)])
                bin_labels = torch.cat([bin_labels, torch.full((pad_len,), -100, dtype=bin_labels.dtype)])
                attention_mask[-pad_len:] = 0

            batch_input_ids.append(bin_ids)
            batch_labels.append(bin_labels)
            batch_attention_mask.append(attention_mask)

        result = {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention_mask),
            "pixel_values": torch.cat(all_pixel_values, dim=0),
            "num_image_tokens": torch.cat(all_num_image_tokens, dim=0),
        }
        return result


@dataclass
class PackedBatchCollator:
    """Collator for ConstantLengthDataset outputs.

    Since packing already produces fixed-length tensors, this collator simply
    stacks them into batch tensors. No padding or further processing needed.
    """

    def __call__(self, features: list[dict[str, Any] | None]) -> dict[str, Any]:
        features = [f for f in features if f is not None]
        if not features:
            raise ValueError("Collator received empty features list!")

        input_ids = torch.stack([f["input_ids"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)
        num_image_tokens = torch.cat([f["num_image_tokens"] for f in features], dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "num_image_tokens": num_image_tokens,
        }


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
