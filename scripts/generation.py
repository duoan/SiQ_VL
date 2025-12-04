#!/usr/bin/env python3
"""
Standalone generation/inference script for SiQ-VL model.
"""

import argparse
import os
import sys

from PIL import Image
from transformers import AutoProcessor

# Add parent directory to path to import siq_vl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siq_vl.model.modeling import SiQ_VLModel


def main():
    """Main function for standalone inference."""
    parser = argparse.ArgumentParser(description="Run inference with SiQ-VL model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask about the image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: auto-detect)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=2,
        help="Number of beams for beam search (default: 2)",
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Disable sampling (use greedy decoding)",
    )

    args = parser.parse_args()

    # Load model and processor
    vl_model = SiQ_VLModel.from_pretrained(args.checkpoint)
    vl_model.to(args.device)
    vl_model.eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    inputs = processor(batch=[(args.question, Image.open(args.image), None)])
    output_ids = vl_model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_beams=args.num_beams,
        do_sample=not args.no_sample,
    )
    answer = processor.batch_decode(output_ids, assistant_only=True, skip_special_tokens=True)
    print(f"\n>>> Answer: {answer}\n")


if __name__ == "__main__":
    main()
