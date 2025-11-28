from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class SiQ_VLDataCollator:
    tokenizer: Any
    padding_side: str = "right"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Filter out None
        features = [f for f in features if f is not None]
        # sort by length to reduce padding waste
        features = sorted(features, key=lambda x: len(x["input_ids"]), reverse=True)

        # --- DEBUG START ---
        # Check the first feature to see what keys exist
        if len(features) > 0:
            keys = features[0].keys()
            # print(f"DEBUG: Collator received keys: {keys}") # Uncomment if needed

            if "pixel_values" not in keys:
                raise ValueError(
                    "CRITICAL: 'pixel_values' is MISSING from Dataset output! Check dataset.py"
                )
        # --- DEBUG END ---

        # 1. Standard Tensor Fields
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]

        # 2. Pixel Values (Stacking)
        pixel_values_list = []
        for i, f in enumerate(features):
            if "pixel_values" in f and f["pixel_values"].numel() > 0:
                pixel_values_list.append(f["pixel_values"])
            else:
                print(
                    f"WARNING: Sample {i} has empty or invalid pixel_values: {type(f['pixel_values'])}"
                )

        # --- CRITICAL FIX ---
        # If we have no images, we MUST NOT return None for pixel_values if the model expects them.
        if len(pixel_values_list) > 0:
            pixel_values_stacked = torch.stack(pixel_values_list, dim=0)
        else:
            # If your dataset guarantees images (which it should), this is an error state.
            raise ValueError(
                f"Collator found 0 valid images in a batch of {len(features)} samples! Check your Processor/Dataset."
            )

        # 3. Padding Input IDs & Labels
        batch_size = len(features)
        max_len = max(len(input_ids) for input_ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id

        input_ids_padded = torch.full(
            (batch_size, max_len),
            fill_value=pad_id,
            dtype=input_ids_list[0].dtype,
        )
        labels_padded = torch.full(
            (batch_size, max_len),
            fill_value=-100,
            dtype=labels_list[0].dtype,
        )

        for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
            length = min(ids.size(0), max_len)
            input_ids_padded[i, :length] = ids[:length]
            labels_padded[i, :length] = labs[:length]

        attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id).long()

        # 4. Construct Batch Dictionary
        batch: Dict[str, Any] = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask,
        }

        if pixel_values_stacked is not None:
            batch["pixel_values"] = pixel_values_stacked

        # 5. HANDLE RAW METADATA (Pass through as lists)
        # Check if the first sample has 'raw_question'
        if "question" in features[0]:
            batch["question"] = [f["question"] for f in features]

        if "answer" in features[0]:
            batch["answer"] = [f["answer"] for f in features]

        if "image" in features[0]:
            batch["image"] = [f["image"] for f in features]

        return batch
