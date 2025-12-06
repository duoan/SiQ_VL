import random

from PIL import Image
from torch.utils.data import Dataset


def _to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    return pil_image.convert("RGB")


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
        images = item.get("images", [])
        texts = item.get("texts", [])

        # Drop no image or text samples
        if len(images) == 0 or len(texts) == 0:
            return None

        if not isinstance(images[0], Image.Image):
            return None

        image = _to_rgb(images[0])

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
