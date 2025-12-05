from dataclasses import dataclass
from typing import Any


@dataclass
class SiQ_VLDataCollator:
    processor: Any
    max_length: int | None = None
    return_raw_data: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
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

        # Call processor to handle tokenization, image processing, and label generation
        # Processor uses padding="longest" by default, which will pad to the longest sequence in the batch
        processed = self.processor(
            batch=batch,
            return_tensors="pt",
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding="longest",  # Pad to longest sequence in batch
        )

        # Processor returns BatchEncoding with input_ids, pixel_values, labels, attention_mask
        result = dict(processed)

        # Add raw metadata if needed
        if self.return_raw_data:
            result["questions"] = questions
            result["answers"] = answers
            result["images"] = images

        return result
