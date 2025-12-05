from collections.abc import Iterator
import random

from PIL import Image
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
        if image.channels != 3:
            return None
        return image

    # Case 2 — It is a file path (string)
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
        if img.channels != 3:
            return None
        return img

    return None


class VQADataset(Dataset):
    """
    Standard Dataset that randomly selects one turn per item on each access.
    Supports DistributedSampler automatically via Trainer's DataLoader.

    This approach is much faster during initialization since we don't need to
    pre-expand all samples. Each item will be visited once per epoch, but a
    random turn will be selected each time. To cover all turns, run multiple
    training epochs.
    """

    def __init__(self, hf_dataset, is_fixed: bool = False):
        """
        hf_dataset: HuggingFace dataset object
        """
        self.dataset = hf_dataset
        self.is_fixed = is_fixed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = _to_pil_rgb(item["images"][0])
        # drop item if image is None
        if image is None:
            return None

        texts = item.get("texts", [])

        if len(texts) == 0:
            # Fallback if no texts
            return {
                "image": image,
                "question": "",
                "answer": "",
            }

        # Randomly select one turn from this item
        # Each epoch will see a different random turn, allowing coverage of all turns
        # across multiple training epochs
        turn_idx = 0 if self.is_fixed else random.randint(0, len(texts) - 1)

        turn = texts[turn_idx]
        q = turn.get("user", "")
        a = turn.get("assistant", "")

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
