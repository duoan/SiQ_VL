#!/usr/bin/env python3
"""
Standalone generation/inference script for SiQ-VL model.
"""

import argparse
import os
import sys

from PIL import Image
import torch

# Add parent directory to path to import siq_vl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siq_vl.model.modeling import SiQ_VLForCausalLM
from siq_vl.model.processing import SiQ_VLProcessor


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
        required=False,
        default=f"{os.path.dirname(__file__)}/../image.png",
        help="Path to input image file",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=False,
        default="What is shown in the image?",
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
        default=64,
        help="Maximum number of new tokens to generate (default: 64)",
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
        default=1,
        help="Number of beams for beam search (default: 1)",
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Disable sampling (use greedy decoding)",
    )

    args = parser.parse_args()

    # Load model and processor
    vl_model = SiQ_VLForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vl_model.to(device)
    vl_model.eval()

    processor = SiQ_VLProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    inputs = processor(batch=[(Image.open(args.image), args.question, None)])
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    output_ids = vl_model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_beams=args.num_beams,
        do_sample=not args.no_sample,
        use_cache=False,
    )
    answer = processor.batch_decode(output_ids, assistant_only=True, skip_special_tokens=True)
    print(f"\n>>> Answer: {answer}\n")


if __name__ == "__main__":
    main()
