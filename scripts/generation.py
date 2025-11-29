#!/usr/bin/env python3
"""
Standalone generation/inference script for SiQ-VL model.
"""

import argparse
import sys
import os

# Add parent directory to path to import siq_vl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siq_vl.model import load_model_from_checkpoint


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
    model, processor = load_model_from_checkpoint(args.checkpoint, device=args.device)
    
    # Generate answer
    print(f"\n>>> Question: {args.question}")
    print(">>> Generating answer...")
    
    answer = model.generate_answer(
        processor=processor,
        samples=(args.image, args.question),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=not args.no_sample,
        num_beams=args.num_beams,
    )
    
    print(f"\n>>> Answer: {answer}\n")


if __name__ == "__main__":
    main()

