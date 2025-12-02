from collections.abc import Iterator

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


def _to_pil_rgb(image: str | Image.Image) -> Image.Image:
    """
    Normalize any image-like input into a valid 3-channel RGB PIL.Image.

    SigLIP expects images in channels-first format (C, H, W). However, we should
    never pass tensors or numpy arrays directly to the model, because unusual
    shapes like (1, 1, 3) confuse the channel inference logic inside the
    image processor. This helper ensures a clean, unambiguous PIL RGB image so
    that SigLIP can safely convert it to (3, H, W) internally.
    """

    # Case 1 — Already a PIL image
    if isinstance(image, Image.Image):
        # Ensure it's RGB (common case: grayscale → RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    # Case 2 — It is a file path (string)
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
        return img

    # Case 3 — torch.Tensor or numpy array
    # Convert to numpy array for unified processing
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.array(image)

    # Ensure the array has at least 3 dimensions
    # Typical formats are:
    # - H x W x C  (channels_last)
    # - C x H x W  (channels_first)
    if arr.ndim == 3:
        # If it's channels-first (C, H, W) → convert to (H, W, C)
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

    # If the array is grayscale (H, W) → duplicate into RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # If the last dimension is single-channel, replicate it 3 times
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # If the last dimension is 4 (RGBA), convert to RGB
    if arr.ndim == 3 and arr.shape[-1] == 4:
        # Remove alpha channel
        arr = arr[:, :, :3]

    # Ensure values are uint8 (required by PIL)
    if arr.dtype != np.uint8:
        # Clip out-of-range values and cast
        arr = np.clip(arr, 0, 255).astype("uint8")

    # Final safety check: ensure we have exactly 3 channels
    if arr.ndim != 3 or arr.shape[-1] != 3:
        # If still not RGB, force convert through PIL
        temp_img = Image.fromarray(arr) if arr.ndim >= 2 else Image.new("RGB", (1, 1))
        return temp_img.convert("RGB")

    # Convert final array into a proper RGB PIL image
    return Image.fromarray(arr, mode="RGB")


class VQADataset(Dataset):
    """
    Standard Dataset that expands multi-turn conversations into individual samples.
    Supports DistributedSampler automatically via Trainer's DataLoader.
    """

    def __init__(self, hf_dataset):
        """
        hf_dataset: HuggingFace dataset object
        """
        self.dataset = hf_dataset
        # Pre-expand all samples: build a list of (item_idx, turn_idx) pairs
        self.samples = []
        for item_idx in range(len(hf_dataset)):
            item = hf_dataset[item_idx]
            num_turns = len(item.get("texts", []))
            for turn_idx in range(num_turns):
                self.samples.append((item_idx, turn_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_idx, turn_idx = self.samples[idx]
        item = self.dataset[item_idx]
        image = _to_pil_rgb(item["images"][0])
        turn = item["texts"][turn_idx]
        q = turn["user"]
        a = turn["assistant"]

        return {
            "image": image,
            "question": q,
            "answer": a,
        }


class VQAIterableDataset(IterableDataset):
    """
    IterableDataset version (for backward compatibility).
    Note: This does NOT support DistributedSampler automatically.
    You need to manually shard the dataset before creating this.
    """

    def __init__(
        self,
        hf_dataset,
    ):
        """
        hf_dataset: HuggingFace dataset object
        """
        self.dataset = hf_dataset

    def __iter__(self) -> Iterator:
        worker = get_worker_info()
        if worker is None:
            start = 0
            step = 1
        else:
            start = worker.id
            step = worker.num_workers

        for i in range(start, len(self.dataset), step):
            item = self.dataset[i]
            image = _to_pil_rgb(item["images"][0])

            for turn in item["texts"]:
                q = turn["user"]
                a = turn["assistant"]

                yield {
                    "image": image,
                    "question": q,
                    "answer": a,
                }
