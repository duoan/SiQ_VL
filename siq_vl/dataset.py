import itertools
import json
import os
import random
import threading
from queue import Queue
from typing import Iterator

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info


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


class ProcessedVQADataset(Dataset):
    """VQADataset that pre-processes samples in __getitem__ (parallelized by DataLoader workers).

    Returns tokenized + image-processed tensors per sample, so the collator
    only needs to do fast bin-packing (no image processing).
    """

    def __init__(self, hf_dataset, processor, max_length: int = 1024, is_fixed: bool = False):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length
        self.is_fixed = is_fixed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        images = item.get("images", [])
        texts = item.get("texts", [])

        if len(images) == 0 or len(texts) == 0:
            return None
        if not isinstance(images[0], Image.Image):
            return None

        image = _to_rgb(images[0])
        turn_idx = 0 if self.is_fixed else random.randint(0, len(texts) - 1)
        turn = texts[turn_idx]
        q = turn.get("user", "")
        a = turn.get("assistant", "")

        if any(keyword.lower() in a.lower() for keyword in _reject_keywords):
            return None

        processed = self.processor(
            batch=[(image, q, a)],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        input_ids = processed["input_ids"].squeeze(0)
        labels = processed["labels"].squeeze(0) if "labels" in processed else torch.full_like(input_ids, -100)
        pixel_values = processed["pixel_values"]  # (n_tiles, C, H, W)
        num_image_tokens = processed["num_image_tokens"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "num_image_tokens": num_image_tokens,
            "length": input_ids.shape[0],
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


class ConstantLengthDataset(IterableDataset):
    """Online packing dataset that yields fixed-length sequences.

    Wraps a VQADataset and a processor to:
    1. Tokenize + process images per sample (no padding)
    2. Buffer samples, then greedily pack them into bins of exactly `seq_length`
    3. Yield fixed-shape tensors -- eliminates TileGym re-autotuning and padding waste

    Inspired by nanoVLM's ConstantLengthDataset.
    """

    def __init__(
        self,
        dataset: VQADataset,
        processor,
        seq_length: int = 1024,
        num_of_sequences: int = 32,
        infinite: bool = True,
        queue_size: int = 8,
        max_images_per_knapsack: int = 8,
    ):
        self.dataset = dataset
        self.processor = processor
        self.seq_length = seq_length
        self.num_of_sequences = num_of_sequences
        self.max_length = seq_length * num_of_sequences
        self.infinite = infinite
        self.queue_size = max(queue_size, 1)
        self.max_images_per_knapsack = max_images_per_knapsack
        self._sentinel = object()
        self.epoch = 0

    def __len__(self):
        avg_sample_len = 300
        return int(len(self.dataset) * avg_sample_len / self.seq_length)

    def __iter__(self) -> Iterator[dict]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        def make_base_iterator():
            all_indices = list(range(len(self.dataset)))
            random.shuffle(all_indices)
            if num_workers > 1:
                worker_indices = itertools.islice(all_indices, worker_id, None, num_workers)
            else:
                worker_indices = iter(all_indices)
            return worker_indices

        queue: Queue = Queue(maxsize=self.queue_size)

        producer = threading.Thread(
            target=self._producer, args=(make_base_iterator, queue), daemon=True
        )
        producer.start()

        while True:
            packed_batch = queue.get()
            if packed_batch is self._sentinel:
                break
            for sample in packed_batch:
                yield sample

    def _producer(self, make_iterator, queue: Queue):
        """Background thread: tokenizes, buffers, packs, and enqueues fixed-length samples."""
        index_iter = make_iterator()
        more_examples = True

        while more_examples:
            buffer = []
            buffer_len = 0

            while buffer_len < self.max_length:
                try:
                    idx = next(index_iter)
                except StopIteration:
                    if self.infinite:
                        index_iter = make_iterator()
                        self.epoch += 1
                        continue
                    else:
                        more_examples = False
                        break

                raw_sample = self.dataset[idx]
                if raw_sample is None:
                    continue

                processed = self._process_single(raw_sample)
                if processed is None:
                    continue

                if processed["length"] > self.seq_length:
                    continue

                buffer.append(processed)
                buffer_len += processed["length"]

            if not buffer:
                break

            groups = self._balanced_greedy_knapsack(buffer)
            packed_samples = []
            for group in groups:
                packed = self._pack_group(group, buffer)
                if packed is not None:
                    packed_samples.append(packed)

            if packed_samples:
                queue.put(packed_samples)

        queue.put(self._sentinel)

    def _process_single(self, raw_sample: dict):
        """Tokenize and process a single sample without padding."""
        try:
            processed = self.processor(
                batch=[(raw_sample["image"], raw_sample["question"], raw_sample["answer"])],
                return_tensors="pt",
                truncation=True,
                max_length=self.seq_length,
                padding=False,
            )
        except Exception:
            return None

        input_ids = processed["input_ids"].squeeze(0)
        labels = processed["labels"].squeeze(0) if "labels" in processed else torch.full_like(input_ids, -100)
        pixel_values = processed["pixel_values"]
        num_image_tokens = processed["num_image_tokens"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "num_image_tokens": num_image_tokens,
            "length": input_ids.shape[0],
            "n_images": pixel_values.shape[0] if pixel_values.dim() >= 1 else 1,
        }

    def _balanced_greedy_knapsack(self, buffer: list) -> list:
        """Pack buffer items into bins of capacity seq_length using balanced greedy knapsack."""
        lengths = [s["length"] for s in buffer]
        image_counts = [s["n_images"] for s in buffer]

        items = sorted(
            enumerate(zip(lengths, image_counts)), key=lambda x: x[1][0], reverse=True
        )

        total_len = sum(lengths)
        min_knapsacks = (total_len + self.seq_length - 1) // self.seq_length + 5
        knapsack_load = [0] * min_knapsacks
        knapsack_images = [0] * min_knapsacks
        knapsack_groups: list[list[int]] = [[] for _ in range(min_knapsacks)]

        for idx, (item_len, item_images) in items:
            suitable = None
            for ks_id in sorted(range(len(knapsack_load)), key=knapsack_load.__getitem__):
                length_fits = knapsack_load[ks_id] + item_len <= self.seq_length
                image_fits = (
                    self.max_images_per_knapsack is None
                    or knapsack_images[ks_id] + item_images <= self.max_images_per_knapsack
                )
                if length_fits and image_fits:
                    suitable = ks_id
                    break

            if suitable is None:
                suitable = len(knapsack_load)
                knapsack_load.append(0)
                knapsack_images.append(0)
                knapsack_groups.append([])

            knapsack_groups[suitable].append(idx)
            knapsack_load[suitable] += item_len
            knapsack_images[suitable] += item_images

        random.shuffle(knapsack_groups)
        return [g for g in knapsack_groups if g]

    def _pack_group(self, group_indices: list, buffer: list) -> dict:
        """Pack a group of samples into a single fixed-length sequence."""
        all_ids = []
        all_labels = []
        all_pixel_values = []
        all_num_image_tokens = []

        for idx in group_indices:
            s = buffer[idx]
            all_ids.append(s["input_ids"])
            all_labels.append(s["labels"])
            all_pixel_values.append(s["pixel_values"])
            all_num_image_tokens.append(s["num_image_tokens"])

        packed_ids = torch.cat(all_ids)
        packed_labels = torch.cat(all_labels)

        if packed_ids.shape[0] > self.seq_length:
            packed_ids = packed_ids[: self.seq_length]
            packed_labels = packed_labels[: self.seq_length]

        pad_len = self.seq_length - packed_ids.shape[0]
        if pad_len > 0:
            pad_token_id = self.processor.tokenizer.pad_token_id or 0
            packed_ids = torch.cat([packed_ids, torch.full((pad_len,), pad_token_id, dtype=packed_ids.dtype)])
            packed_labels = torch.cat([packed_labels, torch.full((pad_len,), -100, dtype=packed_labels.dtype)])

        attention_mask = torch.ones(self.seq_length, dtype=torch.long)
        if pad_len > 0:
            attention_mask[-pad_len:] = 0

        pixel_values = torch.cat(all_pixel_values, dim=0)
        num_image_tokens = torch.cat(
            [t if t.dim() > 0 else t.unsqueeze(0) for t in all_num_image_tokens], dim=0
        )

        return {
            "input_ids": packed_ids,
            "labels": packed_labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "num_image_tokens": num_image_tokens,
        }
