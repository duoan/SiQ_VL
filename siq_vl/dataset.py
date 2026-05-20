import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset


def _to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    return pil_image.convert("RGB")


_reject_keywords = [
    "cannot see",
    "can't see",
    "don't have access",
    "don't have the ability",
    "text-based AI",
    "as an AI",
    "I'm sorry",
    "I apologize",
    "unable to view",
    "cannot view",
    "I'm unable to",
    "I cannot provide",
    "I don't have the capability",
    "as a language model",
    "as a text-based",
    "I do not have access",
]


class VQADataset(Dataset):
    """
    Standard Dataset that randomly selects one turn per item on each access.
    Supports DistributedSampler automatically via Trainer's DataLoader.

    This approach is much faster during initialization since we don't need to
    pre-expand all samples. Each item will be visited once per epoch, but a
    random turn will be selected each time. To cover all turns, run multiple
    training epochs.
    """

    def __init__(self, hf_dataset, is_fixed: bool = False, tokens_per_tile: int = 64):
        """
        hf_dataset: HuggingFace dataset object
        tokens_per_tile: vision tokens per image tile (for length estimation)
        """
        self.dataset = hf_dataset
        self.is_fixed = is_fixed
        self._tokens_per_tile = tokens_per_tile
        self._lengths = None

    @property
    def lengths(self) -> list[int]:
        """Approximate token lengths for LengthGroupedSampler (HF Trainer group_by_length)."""
        if self._lengths is None:
            self._lengths = self._estimate_lengths()
        return self._lengths

    def _estimate_lengths(self) -> list[int]:
        """Fast approximate length estimation using char count heuristic (~3.5 chars/token for English)."""
        lengths = []
        chars_per_token = 3.5
        template_overhead = 80  # system prompt + chat template tokens
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            texts = item.get("texts", [])
            if not texts:
                lengths.append(0)
                continue
            turn = texts[0]
            q = turn.get("user", "")
            a = turn.get("assistant", "")
            text_tokens = int((len(q) + len(a)) / chars_per_token) + template_overhead
            # Estimate vision tokens: assume ~4 tiles average for dynamic tiling
            vision_tokens = 4 * self._tokens_per_tile
            lengths.append(text_tokens + vision_tokens)
        return lengths

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

        # Reject samples with unwanted keywords in the answer
        if any(keyword.lower() in a.lower() for keyword in _reject_keywords):
            return None

        return {
            "image": image,
            "question": q,
            "answer": a,
        }


class CachedVQADataset(Dataset):
    """
    Dataset that uses pre-extracted vision features instead of raw images.
    Skips the vision encoder forward pass entirely during training.

    Expects a cache directory produced by scripts/extract_vision_features.py
    containing shard_*.pt files and a metadata.json.
    """

    def __init__(self, hf_dataset, cache_dir: str, is_fixed: bool = False):
        self.dataset = hf_dataset
        self.is_fixed = is_fixed
        self.cache_dir = cache_dir

        with open(os.path.join(cache_dir, "metadata.json")) as f:
            self.metadata = json.load(f)

        self._shard_cache = {}
        self._shard_size = self.metadata["shard_size"]
        self._num_shards = self.metadata["num_shards"]

    def _load_shard(self, shard_idx: int) -> dict:
        if shard_idx not in self._shard_cache:
            path = os.path.join(self.cache_dir, f"shard_{shard_idx:05d}.pt")
            self._shard_cache[shard_idx] = torch.load(path, weights_only=False, map_location="cpu")
        return self._shard_cache[shard_idx]

    def _get_cached_features(self, idx: int):
        shard_idx = idx // self._shard_size
        shard = self._load_shard(shard_idx)
        if idx not in shard:
            return None
        return shard[idx]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        texts = item.get("texts", [])

        if len(texts) == 0:
            return None

        cached = self._get_cached_features(idx)
        if cached is None:
            return None

        turn_idx = 0 if self.is_fixed else random.randint(0, len(texts) - 1)
        turn = texts[turn_idx]
        q = turn.get("user", "")
        a = turn.get("assistant", "")

        if any(keyword.lower() in a.lower() for keyword in _reject_keywords):
            return None

        return {
            "vision_features": cached["vision_features"],  # (num_tiles, seq_len, hidden_dim)
            "num_tiles": cached["num_tiles"],
            "question": q,
            "answer": a,
        }
