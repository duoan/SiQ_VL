"""
Offline SigLIP feature extraction for SiQ-VL training.

Pre-computes vision encoder hidden states and saves them alongside
the original dataset index. During training, the model can load these
cached features and skip the entire vision forward pass (428M params).

The cached features are the *pre-projector* outputs:
  shape = (num_tiles, seq_len, hidden_dim) = (N, 1024, 1152)

Usage:
    python scripts/extract_vision_features.py \
        --data_path HuggingFaceM4/FineVision \
        --sub_sets "sharegpt4v(coco)" \
        --output_dir data/cached_features/sharegpt4v_coco \
        --batch_size 32
"""

import argparse
import os
import time

import torch
import torch.cuda
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.prints import rank_zero_info
from tqdm import tqdm

from siq_vl.dataset import _to_rgb
from siq_vl.model.modeling import SiQ_VLVisionModel
from siq_vl.model.processing import SiQ_VLImageProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SigLIP vision features offline")
    parser.add_argument("--vision_model_name_or_path", type=str, default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--data_path", type=str, default="HuggingFaceM4/FineVision")
    parser.add_argument("--sub_sets", type=str, default="sharegpt4v(coco)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="data/cached_features")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pixel_shuffle_factor", type=int, default=4)
    parser.add_argument("--shard_size", type=int, default=1000, help="Samples per shard file")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class ImageExtractionDataset(Dataset):
    """Wraps HF dataset to yield (index, processed_tiles, num_tiles) for extraction."""

    def __init__(self, hf_dataset, image_processor: SiQ_VLImageProcessor):
        self.dataset = hf_dataset
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        images = item.get("images", [])

        if not images or not isinstance(images[0], Image.Image):
            return None

        image = _to_rgb(images[0])
        result = self.image_processor([image], enable_dynamic_tiling=True, return_tensors="pt")
        pixel_values = result["pixel_values"]  # (num_tiles, 3, H, W)
        num_tiles = result["num_tiles_per_image"][0].item()

        return {
            "idx": idx,
            "pixel_values": pixel_values,
            "num_tiles": num_tiles,
        }


def collate_extraction(batch):
    """Custom collator that handles variable tile counts per sample."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    indices = [b["idx"] for b in batch]
    num_tiles = [b["num_tiles"] for b in batch]
    pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)

    return {
        "indices": indices,
        "pixel_values": pixel_values,
        "num_tiles": num_tiles,
    }


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    rank_zero_info("=" * 80)
    rank_zero_info("SiQ-VL OFFLINE VISION FEATURE EXTRACTION")
    rank_zero_info("=" * 80)
    rank_zero_info(f"Vision model: {args.vision_model_name_or_path}")
    rank_zero_info(f"Data: {args.data_path} / {args.sub_sets}")
    rank_zero_info(f"Output: {args.output_dir}")
    rank_zero_info(f"Batch size: {args.batch_size}")
    rank_zero_info(f"Dtype: {args.dtype}")
    rank_zero_info(f"Shard size: {args.shard_size}")
    rank_zero_info("=" * 80)

    # 1. Load vision model
    rank_zero_info(">>> Loading vision model...")
    vision_model = SiQ_VLVisionModel.from_pretrained(
        args.vision_model_name_or_path, torch_dtype=dtype, attn_implementation="sdpa"
    )
    vision_model = vision_model.to(device).eval()
    rank_zero_info(f">>> Vision model loaded: {sum(p.numel() for p in vision_model.parameters()) / 1e6:.1f}M params")

    # 2. Load dataset
    rank_zero_info(">>> Loading dataset...")
    sub_sets = [s.strip() for s in args.sub_sets.split(",")]
    raw_datasets = []
    for subset in sub_sets:
        ds = load_dataset(args.data_path, name=subset, split="train", num_proc=args.num_proc)
        raw_datasets.append(ds)

    if len(raw_datasets) > 1:
        from datasets.combine import interleave_datasets
        raw_dataset = interleave_datasets(raw_datasets, seed=args.seed, stopping_strategy="first_exhausted")
    else:
        raw_dataset = raw_datasets[0]

    if args.max_samples:
        raw_dataset = raw_dataset.select(range(min(args.max_samples, len(raw_dataset))))

    rank_zero_info(f">>> Dataset size: {len(raw_dataset)}")

    # 3. Create image processor (same as used by SiQ_VLProcessor)
    image_processor = SiQ_VLImageProcessor(
        vit_image_size=vision_model.config.image_size,
    )

    extraction_dataset = ImageExtractionDataset(raw_dataset, image_processor)
    dataloader = DataLoader(
        extraction_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_extraction,
        pin_memory=True,
        drop_last=False,
    )

    # 4. Extract features
    rank_zero_info(">>> Extracting features...")
    all_features = {}
    total_tiles = 0
    start_time = time.time()

    for batch in tqdm(dataloader, desc="Extracting"):
        if batch is None:
            continue

        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
        outputs = vision_model(pixel_values)
        hidden_states = outputs.last_hidden_state  # (total_tiles_in_batch, seq_len, hidden_dim)

        # Split back by sample using num_tiles
        offset = 0
        for i, (idx, n_tiles) in enumerate(zip(batch["indices"], batch["num_tiles"])):
            features = hidden_states[offset:offset + n_tiles].cpu()  # (n_tiles, 1024, 1152)
            all_features[idx] = features
            offset += n_tiles
            total_tiles += n_tiles

    elapsed = time.time() - start_time
    rank_zero_info(f">>> Extraction done: {len(all_features)} samples, {total_tiles} tiles in {elapsed:.1f}s")
    rank_zero_info(f">>> Throughput: {total_tiles / elapsed:.1f} tiles/s")

    # 5. Save as sharded .pt files
    rank_zero_info(f">>> Saving to {args.output_dir}...")
    sorted_indices = sorted(all_features.keys())
    shard_idx = 0
    shard_data = {}

    for i, idx in enumerate(sorted_indices):
        shard_data[idx] = {
            "vision_features": all_features[idx],  # (n_tiles, seq_len, hidden_dim)
            "num_tiles": all_features[idx].shape[0],
        }

        if len(shard_data) >= args.shard_size or i == len(sorted_indices) - 1:
            shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt")
            torch.save(shard_data, shard_path)
            rank_zero_info(f"    Saved {shard_path} ({len(shard_data)} samples)")
            shard_data = {}
            shard_idx += 1

    # 6. Save metadata
    metadata = {
        "vision_model": args.vision_model_name_or_path,
        "dtype": args.dtype,
        "total_samples": len(all_features),
        "total_tiles": total_tiles,
        "hidden_dim": hidden_states.shape[-1],
        "seq_len_per_tile": hidden_states.shape[-2],
        "pixel_shuffle_factor": args.pixel_shuffle_factor,
        "num_shards": shard_idx,
        "shard_size": args.shard_size,
        "extraction_time_s": elapsed,
        "data_path": args.data_path,
        "sub_sets": args.sub_sets,
    }
    import json
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    rank_zero_info(f">>> Metadata saved to {meta_path}")

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    rank_zero_info(f">>> Peak VRAM: {peak_vram:.2f} GB")
    rank_zero_info(">>> Done!")


if __name__ == "__main__":
    main()
