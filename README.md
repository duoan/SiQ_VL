# SiQ-VL: A Vision-Language Model for Multimodal Understanding

[![GitHub](https://img.shields.io/badge/GitHub-duoan/SiQ_VL-black?logo=github&style=flat-square)](https://github.com/duoan/SiQ_VL) [![W&B](https://img.shields.io/badge/W%26B-Metrics_-yellow)](https://wandb.ai/ReproduceAI/siq-vl) [![Hugging Face Stage1](https://img.shields.io/badge/HF-stage1-orange?logo=huggingface&style=flat-square)](https://huggingface.co/classtag/siq-vl_siglip2-large-patch16-512_qwen2.5-1.5b-instruct_stage1) [![Hugging Face Stage2](https://img.shields.io/badge/HF-stage2-orange?logo=huggingface&style=flat-square)](https://huggingface.co/classtag/siq-vl_siglip2-large-patch16-512_qwen2.5-1.5b-instruct_stage2) [![Tech Report](https://img.shields.io/badge/Tech%20Report-PDF-red?logo=readthedocs&style=flat-square)](docs/report/main.pdf) [![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b?logo=arxiv&style=flat-square)](https://arxiv.org/abs/TBD)

## Abstract

SiQ-VL is a vision-language model (VLM) that integrates a SigLIP-based vision encoder with a Qwen2.5 language model through a learnable projection module. The architecture employs a multi-stage training paradigm designed to progressively develop capabilities in multimodal understanding and text generation tasks.

This repository also serves as a **single-GPU training efficiency recipe** — documenting 15 iterative optimizations that achieve **22.9x speedup** (4,674 → 107,367 tok/s) on an NVIDIA RTX PRO 6000 Blackwell GPU.

## Highlights

- **22.9x training speedup** through systematic optimization (BF16 fix, TileGym cuTile kernels, sequence packing, batch scaling)
- **50–64% VRAM reduction** via grad-in-forward fused cross-entropy (logits never materialized)
- **Full technical report** with 15 experiments (including 10 documented failures): [`docs/report/main.pdf`](docs/report/main.pdf)
- **Reproducible benchmarks**: all scripts in `scripts/benchmark_*.py`

## Experiment Tracking

Training runs and experiments are tracked using Weights & Biases. View training metrics, model checkpoints, and experiment logs at: [https://wandb.ai/ReproduceAI/siq-vl](https://wandb.ai/ReproduceAI/siq-vl)

## Training Efficiency

### Technical Report

The full optimization journey is documented as a LaTeX technical report suitable for arXiv submission:

- **PDF**: [`docs/report/main.pdf`](docs/report/main.pdf)
- **Source**: [`docs/report/main.tex`](docs/report/main.tex)

### Summary of Optimizations

| Iteration | Optimization | Speedup | Key Insight |
|---|---|---|---|
| 0 | Baseline (FP32 bug) | 1.0x | Profiler revealed FP32 GEMM despite `bf16=True` |
| 1 | BF16 dtype + `no_grad` | 3.07x | 2-line fix, largest single gain |
| 3 | Batch size 4→16 | 4.34x | GPU underutilized at small B |
| 7 | `torch.compile` | 5.85x | Replaces Liger (mutually exclusive) |
| 11 | TileGym FA4 | 6.5x | Blackwell-native cuTile attention |
| 13 | TileGym full stack + fused CE | 14.6x | 30–41% faster, 50–64% less VRAM vs Liger |
| 15 | Packing + TileGym (verified) | **22.9x** | 107K Real tok/s on real data |

### Failed Experiments (documented in report)

- Liger-Kernel in Stage 1 (speed regression, frozen LLM has no backward)
- Vision feature caching (SigLIP too fast to benefit, +27% VRAM)
- Custom Triton Flash Attention (cannot emit wgmma/TMA on Blackwell)
- FlexAttention with variable shapes (torch.compile recompilation storm)
- cudagraphs `reduce-overhead` mode (dynamic VLM shapes)
- Batch scaling beyond saturation (flat throughput at MFU ceiling)

### Iteration Log

The raw per-iteration log (Hypothesis → Change → Measurement → Result → Decision) lives in [`docs/training_efficiency_plan.md`](docs/training_efficiency_plan.md).

## Architecture Overview

The SiQ-VL architecture comprises three principal components:

1. **Vision Encoder**: A SigLIP-based vision tower that remains frozen throughout the training process
2. **Projection Module**: A learnable projector that transforms vision features into the language model embedding space, incorporating pixel shuffle operations for sequence length compression
3. **Language Model**: A Qwen2.5 transformer-based model responsible for text generation, which remains frozen in Stage 1 and is fine-tuned in subsequent training stages

### Architectural Diagram

<details>
<summary>Model Architecture Diagram (Mermaid)</summary>

```mermaid
graph TB
    Image[Input Image] --> IP[Image Processor<br/>SigLIP]
    Text[Text Prompt] --> Tokenizer[Tokenizer<br/>Qwen2.5]
    
    IP --> Vision[Vision Tower<br/>SigLIP<br/>🔒 FROZEN]
    Tokenizer --> TextEmb[Text Embeddings]
    
    Vision --> VisionFeat[Vision Features<br/>729×1152]
    VisionFeat --> PixelShuffle[Pixel Shuffle<br/>Factor=3]
    PixelShuffle --> Proj[Linear Projection<br/>10368→896]
    Proj --> Norm[LayerNorm]
    Norm --> VisionEmb[Vision Embeddings<br/>81×896]
    
    VisionEmb --> Fusion[Embedding Fusion<br/>Splice Image Tokens]
    TextEmb --> Fusion
    
    Fusion --> LLM[Language Model<br/>Qwen2.5<br/>🔒 Stage1 / ✅ Stage2+]
    LLM --> Output[Generated Text]
    
    style Vision fill:#ffcccc
    style LLM fill:#ccffcc
    style PixelShuffle fill:#ffffcc
    style Proj fill:#ffffcc
    style Norm fill:#ffffcc
```

</details>

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SiQ-VL Model Architecture                         │
└─────────────────────────────────────────────────────────────────────────────┘

    Input Image                    Text Prompt
         │                              │
         │                              │
         ▼                              ▼
    ┌─────────┐                   ┌──────────────┐
    │  Image  │                   │   Tokenizer  │
    │  (PIL)  │                   │   (Qwen2.5)  │
    └────┬────┘                   └───────┬──────┘
         │                                │
         │                                │
         ▼                                ▼
┌────────────────┐                  ┌──────────────┐
│  Image         │                  │  Text Tokens │
│  Processor     │                  │  + Special   │
│  (SigLIP)      │                  │  Tokens      │
└────┬───────────┘                  └──────┬───────┘
     │                                     │
     │                                     │
     ▼                                     │
┌──────────────────────────────────────────┴──────────────────────────────────┐
│                         Vision Tower (SigLIP)                               │
│                         [FROZEN - All Stages]                               │
│                                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │  Patch   │→ │  Patch   │→ │  Patch   │→ │  Patch   │→ ...                │
│  │ Embedding│  │ Embedding│  │ Embedding│  │ Embedding│                     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                     │
│                                                                             │
│  Output: [Batch, 729, 1152]  (for 384×384 image, patch_size=14)             │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Projector (SiQ_VLModalityProjector)                      │
│                    [TRAINABLE - All Stages]                                 │
│                                                                             │
│  ┌────────────────────────────────────────────────────┐                     │
│  │         Pixel Shuffle (Factor=2, default)          │                     │
│  │  [729, 1152] → Reshape → [182, 4608]               │                     │
│  └────────────────────┬───────────────────────────────┘                     │
│                       │                                                     │
│                       ▼                                                     │
│  ┌────────────────────────────────────────────────────┐                     │
│  │         MLP (Linear Projection)                    │                     │
│  │  [182, 4608] → Linear(4608, 896) → [182, 896]      │                     │
│  └────────────────────┬───────────────────────────────┘                     │
│                                                                             │
│  Output: [Batch, 182, 896]  (compressed vision tokens, factor=2 example)    │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     │  ┌───────────────────┐
                                     │  │  Text Embeddings  │
                                     │  │  [Batch, Seq, 896]│
                                     │  └────────┬──────────┘
                                     │           │
                                     ▼           ▼
                              ┌─────────────────────────┐
                              │   Embedding Fusion      │
                              │   (Splice Image Tokens) │
                              └────────────┬────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Language Model (Qwen2.5)                                  │
│                    [FROZEN - Stage 1] [TRAINABLE - Stage 2+]                 │
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │  Layer 1 │→ │  Layer 2 │→ │  Layer 3 │→ │  Layer N │→ ...                 │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                      │
│                                                                              │
│  Output: [Batch, Seq, Vocab]  (logits for next token prediction)             │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │  Generated   │
                              │    Text      │
                              └──────────────┘

Key Dimensions (example with pixel_shuffle_factor=2):
  • Vision Features: [Batch, 729, 1152]  (SigLIP2, 384×384 image, patch_size=14)
  • After Pixel Shuffle: [Batch, 182, 4608]  (729/4 = 182, 1152×4 = 4608)
  • After Projection: [Batch, 182, 896]   (Qwen2.5-0.5B hidden size)
  • LLM Output: [Batch, Seq, Vocab]
```

### Forward Pass Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Forward Pass Data Flow                               │
└─────────────────────────────────────────────────────────────────────────────┘

Input:
  • Image: PIL.Image (384×384×3)
  • Text: "Describe this image."

Step 1: Image Processing
  Image (384×384×3)
    ↓ [Image Processor]
  Pixel Values [1, 3, 384, 384]
    ↓ [Vision Tower - SigLIP]
  Vision Features [1, 729, 1152]
    │
    ├─ 729 patches = (384/14)²
    └─ 1152 = SigLIP SO400M hidden size

Step 2: Projection with Pixel Shuffle
  Vision Features [1, 729, 1152]
    ↓ [Reshape: 27×27 patches]
  [1, 27, 27, 1152]
    ↓ [Pixel Shuffle: factor=2 (default)]
  [1, 13, 13, 4608]  (1152 × 2² = 4608, rounded to 13×13)
    ↓ [Reshape]
  [1, 169, 4608]
    ↓ [MLP: Linear(4608, 896)]
  Vision Embeddings [1, 169, 896]
    │
    ├─ 169 tokens (compressed from 729, factor=2)
    └─ 896 = Qwen2.5-0.5B hidden size

Step 3: Text Processing
  Text: "Describe this image."
    ↓ [Tokenizer + Chat Template]
  Input IDs: [151644, 77091, 198, ..., 151655, ..., 151645]
    │
    ├─ <|im_start|>user\n
    ├─ <|vision_start|><|image_pad|>×169<|vision_end|>
    ├─ Describe this image.
    └─ <|im_end|>
    ↓ [Text Embeddings]
  Text Embeddings [1, Seq, 896]

Step 4: Embedding Fusion
  Text Embeddings: [1, Seq, 896]
    │
    └─ Find <|image_pad|> positions
       │
       ├─ Prefix: [1, prefix_len, 896]
       ├─ Image:  [1, 169, 896]  ← Insert here
       └─ Suffix: [1, suffix_len, 896]
    ↓ [Concatenate]
  Fused Embeddings [1, prefix_len + 169 + suffix_len, 896]

Step 5: LLM Forward Pass
  Fused Embeddings [1, Total_Seq, 896]
    ↓ [Qwen2.5 Transformer]
  Logits [1, Total_Seq, Vocab_Size]
    ↓ [Generate/Decode]
  Output: "The image depicts a beautiful sunset..."

Step 6: Loss Calculation (Training)
  Logits [1, Total_Seq, Vocab_Size]
    │
    └─ Labels [1, Total_Seq]
       │
       ├─ -100 (ignore): Image tokens, prompt tokens
       └─ Token IDs: Answer tokens only
    ↓ [Cross Entropy Loss]
  Loss: scalar
```

### Component Status by Stage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Component Training Status by Stage                       │
└─────────────────────────────────────────────────────────────────────────────┘

Component          │ Stage 1 │ Stage 2 │ Stage 3 │ Stage 4 │
───────────────────┼─────────┼─────────┼─────────┼─────────┤
Vision Tower       │ Frozen  │ Frozen  │ Frozen  │ Frozen  │
(SigLIP)           │         │         │         │         │
───────────────────┼─────────┼─────────┼─────────┼─────────┤
Projector          │ Train   │ Train   │ Train   │ Train   │
                   │         │         │         │         │
───────────────────┼─────────┼─────────┼─────────┼─────────┤
Language Model     │ Frozen  │ Train   │ Train   │ Train   │
(Qwen2.5)          │         │         │         │         │
───────────────────┼─────────┼─────────┼─────────┼─────────┤
RL Components      │  N/A    │  N/A    │  N/A    │ Active  │
                   │         │         │         │         │
```

### Key Design Features

- **Multi-Stage Training Paradigm**: A progressive training strategy that transitions from projector alignment to comprehensive model fine-tuning
- **Pixel Shuffle Compression**: Implements spatial compression to reduce vision token sequence length, improving computational efficiency
- **Automatic Configuration**: Dynamically computes pixel shuffle factors based on vision encoder specifications
- **Distributed Training Support**: Facilitates multi-GPU training through the Accelerate framework
- **Memory Optimization**: Incorporates gradient checkpointing and optimized data loading strategies

## Training Methodology

The SiQ-VL model is trained using a multi-stage approach designed to incrementally develop vision-language capabilities:

### Stage 1: Projector Alignment

**Objective**: Establish alignment between vision encoder outputs and the language model embedding space through supervised training of the projection module exclusively.

- **Frozen Components**: Vision encoder (SigLIP) and language model (Qwen2.5)
- **Trainable Parameters**: Projection module only
- **Training Dataset**: FineVision multimodal instruction-following dataset
- **Purpose**: Initialize vision-language feature alignment
- **Implementation Status**: Fully implemented

### Stage 2: Language Model Fine-tuning on Visual Question Answering

**Objective**: Fine-tune the language model component on large-scale visual question answering datasets to enhance visual comprehension and reasoning capabilities.

- **Frozen Components**: Vision encoder (SigLIP)
- **Trainable Parameters**: Projection module and language model (supports LoRA or full fine-tuning)
- **Training Dataset**: FineVision dataset (can be extended to VQAv2, GQA, TextVQA)
- **Purpose**: Develop enhanced visual understanding and question-answering capabilities
- **Implementation Status**: Fully implemented
- **LoRA Support**: Optional LoRA fine-tuning for efficient training (recommended)

### Stage 3: Supervised Fine-tuning with Chain-of-Thought Reasoning

**Objective**: Fine-tune the model on reasoning datasets annotated with chain-of-thought (CoT) demonstrations to improve step-by-step reasoning and explanatory capabilities.

- **Frozen Components**: Vision encoder (SigLIP)
- **Trainable Parameters**: Projection module and language model
- **Training Dataset**: Visual reasoning datasets with chain-of-thought annotations
- **Purpose**: Develop systematic reasoning and step-by-step explanation capabilities
- **Implementation Status**: Planned for future release

### Stage 4: Reinforcement Learning-based Optimization

**Objective**: Enhance model performance through reinforcement learning techniques, such as reinforcement learning from human feedback (RLHF) or direct preference optimization (DPO), to better align outputs with human preferences.

- **Training Method**: Reinforcement learning-based optimization (specific methodology to be determined)
- **Purpose**: Improve output quality and alignment with human preferences
- **Implementation Status**: Planned for future release

### Training Pipeline Flow Diagram

<details>
<summary>Training Pipeline Visualization (Mermaid)</summary>

```mermaid
graph TD
    Start[Initialize Models<br/>SigLIP + Qwen2.5] --> Stage1[Stage 1: Projector Alignment ✅]
    
    Stage1 --> |Train Projector Only| S1Checkpoint[Checkpoint: Stage 1<br/>Aligned Projector]
    
    S1Checkpoint --> Stage2[Stage 2: LLM Fine-tuning ✅]
    Stage2 --> |Train Projector + LLM| S2Checkpoint[Checkpoint: Stage 2<br/>VQA Capable]
    
    S2Checkpoint --> Stage3[Stage 3: SFT with CoT 🚧]
    Stage3 --> |Train Projector + LLM| S3Checkpoint[Checkpoint: Stage 3<br/>Reasoning Capable]
    
    S3Checkpoint --> Stage4[Stage 4: RL Training 🚧]
    Stage4 --> |RL Optimization| Final[Final Model<br/>Production Ready]
    
    Stage1 -.->|Dataset: FineVision| D1[FineVision<br/>Multimodal Instructions]
    Stage2 -.->|Dataset: VQA| D2[VQAv2, GQA, TextVQA]
    Stage3 -.->|Dataset: CoT| D3[Reasoning with CoT]
    Stage4 -.->|Dataset: Preferences| D4[Human Preferences]
    
    style Stage1 fill:#90EE90
    style Stage2 fill:#90EE90
    style Stage3 fill:#FFD700
    style Stage4 fill:#FFD700
    style Final fill:#87CEEB
```

</details>

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline Overview                          │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Initialization                                                     │
    │  • Load SigLIP (frozen)                                             │
    │  • Load Qwen2.5 (frozen)                                            │
    │  • Initialize Projector (random weights)                            │
    └───────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STAGE 1: Projector Alignment  [IMPLEMENTED]                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Vision Tower: FROZEN                                               │
    │  Projector: TRAINABLE                                               │
    │  LLM: FROZEN                                                        │
    │                                                                     │
    │  Dataset: FineVision                                                │
    │  • Multimodal instruction-following                                 │
    │  • ~10 subsets (coco_colors, sharegpt4v, etc.)                      │
    │                                                                     │
    │  Training:                                                          │
    │  • Learning Rate: 1e-3                                              │
    │  • Steps: ~1000                                                     │
    │  • Objective: Align vision features with LLM space                  │
    └───────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │  Checkpoint: Stage 1      │
                    │  • Aligned Projector      │
                    │  • Frozen Vision + LLM    │
                    └───────────────┬───────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STAGE 2: LLM Fine-tuning on VQA  [IMPLEMENTED]                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Vision Tower: FROZEN                                               │
    │  Projector: TRAINABLE (continue from Stage 1)                       │
    │  LLM: TRAINABLE (unfrozen, supports LoRA or full fine-tuning)      │
    │                                                                     │
    │  Dataset: FineVision (can be extended to VQAv2, GQA, TextVQA)     │
    │  • Large-scale multimodal instruction-following                     │
    │  • Focus on visual question answering and understanding            │
    │                                                                     │
    │  Training:                                                          │
    │  • Learning Rate: 2e-5 (lower for LLM)                             │
    │  • Steps: Auto-calculated from max_samples and batch size          │
    │  • Objective: Improve VQA capabilities                             │
    │  • LoRA: Optional efficient fine-tuning (recommended)              │
    └───────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │  Checkpoint: Stage 2      │
                    │  • VQA-capable model      │
                    └───────────────┬───────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STAGE 3: SFT with CoT Reasoning [PLANNED]                          │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Vision Tower: FROZEN                                               │
    │  Projector: TRAINABLE (continue from Stage 2)                       │
    │  LLM: TRAINABLE (continue from Stage 2)                             │
    │                                                                     │
    │  Dataset: Reasoning with Chain-of-Thought                           │
    │  • Step-by-step reasoning annotations                               │
    │  • Visual reasoning tasks                                           │
    │                                                                     │
    │  Training:                                                          │
    │  • Learning Rate: 1e-5 to 2e-5                                      │
    │  • Steps: TBD                                                       │
    │  • Objective: Develop reasoning capabilities                        │
    └───────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │  Checkpoint: Stage 3      │
                    │  • Reasoning-capable      │
                    └───────────────┬───────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STAGE 4: Reinforcement Learning [PLANNED]                          │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Vision Tower: FROZEN                                               │
    │  Projector: TRAINABLE (continue from Stage 3)                       │
    │  LLM: TRAINABLE (continue from Stage 3)                             │
    │  RL Components: ACTIVE                                              │
    │                                                                     │
    │  Dataset: Preference Datasets                                       │
    │  • Human feedback data                                              │
    │  • Preference pairs                                                 │
    │                                                                     │
    │  Training:                                                          │
    │  • Method: RLHF / DPO / etc. (TBD)                                  │
    │  • Objective: Align with human preferences                          │
    └───────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │  Final Model              │
                    │  • Fully aligned VLM      │
                    │  • Production ready       │
                    └───────────────────────────┘
```

### Training Stage Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Training Stage Comparison Table                        │
└─────────────────────────────────────────────────────────────────────────────┘

Feature              │ Stage 1        │ Stage 2        │ Stage 3        │ Stage 4
─────────────────────┼────────────────┼────────────────┼────────────────┼────────────
Status               │ Implemented    │ Implemented    │ Planned        │ Planned
─────────────────────┼────────────────┼────────────────┼────────────────┼────────────
Trainable Components │ Projector only │ Projector+LLM  │ Projector+LLM  │ Projector+LLM+RL
                     │                │ (LoRA/Full)    │                │
─────────────────────┼────────────────┼────────────────┼────────────────┼────────────
Frozen Components    │ Vision + LLM   │ Vision only    │ Vision only    │ Vision only
─────────────────────┼────────────────┼────────────────┼────────────────┼────────────
Learning Rate        │ 1e-3           │ 2e-5           │ 1e-5 to 2e-5   │ TBD
─────────────────────┼────────────────┼────────────────┼────────────────┼────────────
Training Steps       │ ~1000          │ Auto-calc      │ TBD            │ TBD
─────────────────────┼────────────────┼────────────────┼────────────────┼────────────
Primary Dataset      │ FineVision     │ FineVision     │ CoT Reasoning  │ Preferences
─────────────────────┼────────────────┼────────────────┼────────────────┼────────────
Objective            │ Alignment      │ VQA            │ Reasoning      │ Alignment
─────────────────────┼────────────────┼────────────────┼────────────────┼────────────
Checkpoint Input     │ Base models    │ Stage 1        │ Stage 2        │ Stage 3
─────────────────────┼────────────────┼────────────────┼────────────────┼────────────
Checkpoint Output    │ Stage 1        │ Stage 2        │ Stage 3        │ Final Model
```

## Requirements

### System Requirements

- Python 3.10 (Python >= 3.10 and < 3.11)
- PyTorch >= 2.9.1
- CUDA-capable GPU with at least 24GB VRAM (recommended for training)
- Tested on: NVIDIA RTX PRO 6000 Blackwell (96GB), CUDA 13.0
- Package manager: [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Installation

### Installation via uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd SiQ_VL

# Install dependencies
uv sync
```

### Using pip

```bash
pip install -e .
```

## Training Datasets

### Stage 1: FineVision Dataset

Stage 1 training employs the FineVision dataset, available through HuggingFace, which comprises multiple data subsets:

- `coco_colors`
- `densefusion_1m`
- `face_emotion`
- `google_landmarks`
- `laion_gpt4v`
- `sharegpt4o`
- `sharegpt4v(coco)`
- `sharegpt4v(llava)`
- `sharegpt4v(knowledge)`
- `sharegpt4v(sam)`

### Future Training Stages

- **Stage 2**: Large-scale visual question answering datasets (VQAv2, GQA, TextVQA)
- **Stage 3**: Visual reasoning datasets annotated with chain-of-thought demonstrations
- **Stage 4**: Human preference datasets for reinforcement learning optimization

## Training Instructions

> **Note**: Stage 1 (Projector Alignment) and Stage 2 (LLM Fine-tuning) are fully implemented. Stages 3-4 are planned for future releases.

### Stage 1: Projector Alignment Training

#### Quick Start

The easiest way to start Stage 1 training is using the provided shell script, which auto-detects your environment:

```bash
bash scripts/train_stage_1.sh
```

The script performs the following automatic configurations:
- Detects the computing environment (e.g., MacBook, AWS p4d instances)
- Sets appropriate hyperparameters for Stage 1 training
- Configures distributed training when multiple GPUs are available
- Freezes the language model and trains only the projection module

#### Manual Training

For more control, you can run the training script directly:

```bash
python scripts/train.py \
    --vision_model_name_or_path "google/siglip2-base-patch16-224" \
    --text_model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
    --data_path "HuggingFaceM4/FineVision" \
    --sub_sets "coco_colors,densefusion_1m,sharegpt4v(knowledge)" \
    --freeze_text_model \
    --output_dir "./checkpoints" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --learning_rate 1e-3 \
    --bf16
```

**Important**: Stage 1 training employs `--freeze_text_model` by default, ensuring that only the projection module parameters are updated during this training phase.

### Stage 2: Language Model Fine-tuning

#### Quick Start

The easiest way to start Stage 2 training is using the provided shell script:

```bash
bash scripts/train_stage_2.sh
```

This script automatically:
- Loads the Stage 1 checkpoint
- Unfreezes the text model for fine-tuning
- Configures appropriate hyperparameters for Stage 2 training
- Supports LoRA fine-tuning (recommended for efficiency)

#### Manual Training

For more control, you can run Stage 2 training directly:

```bash
python scripts/train.py \
    --stage_1_checkpoint_path "./checkpoints/siq-vl_{vision}_{text}_{datetime}/stage1" \
    --no_freeze_text_model \
    --use_lora \
    --data_path "HuggingFaceM4/FineVision" \
    --sub_sets "coco_colors,densefusion_1m,sharegpt4v(knowledge)" \
    --output_dir "./checkpoints" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --bf16
```

**Important**: Stage 2 training requires a Stage 1 checkpoint. The `--stage_1_checkpoint_path` can be auto-inferred from the model names if not specified. Use `--use_lora` for efficient fine-tuning or omit it for full fine-tuning.

### Training Arguments

#### Model Configuration
- `--vision_model_name_or_path`: Path or HuggingFace model ID for vision encoder (default: `google/siglip2-base-patch16-224`)
- `--text_model_name_or_path`: Path or HuggingFace model ID for language model (default: `Qwen/Qwen2.5-0.5B-Instruct`)
- `--freeze_text_model`: Freeze the text model during training (default: True for Stage 1)
- `--no_freeze_text_model`: Unfreeze the text model for full fine-tuning (Stage 2)
- `--stage_1_checkpoint_path`: Path to Stage 1 checkpoint for Stage 2 training (default: auto-inferred)
- `--pixel_shuffle_factor`: Pixel shuffle factor for the projector (default: 2)
- `--use_lora`: Use LoRA for efficient fine-tuning (default: False, recommended for Stage 2)
- `--lora_r`: LoRA rank (default: 64)
- `--lora_alpha`: LoRA alpha parameter (default: 16)
- `--lora_dropout`: LoRA dropout rate (default: 0.05)
- `--lora_target_modules`: Target modules for LoRA (default: None, auto-detected)

#### Dataset Configuration
- `--data_path`: Path to dataset or HuggingFace dataset name (default: `HuggingFaceM4/FineVision`)
- `--sub_sets`: Comma-separated list of dataset subsets to use
- `--sub_sets_weights`: Optional comma-separated sampling weights aligned with sub_sets (e.g., `"4,4,1,1"`)
- `--max_samples`: Limit dataset size for quick testing
- `--num_proc`: Number of processes for dataset loading (default: 96)
- `--dataloader_num_workers`: Number of dataloader workers (default: 4)

#### Training Hyperparameters
- `--per_device_train_batch_size`: Batch size per device (default: 8)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `--max_steps`: Maximum training steps (default: -1, auto-calculated from max_samples and batch size)
- `--learning_rate`: Learning rate (default: 1e-3 for Stage 1, 2e-5 for Stage 2)
- `--bf16`: Use bfloat16 precision (default: True, recommended for Qwen)
- `--fp16`: Use float16 precision (alternative to bf16)
- `--no_bf16`: Disable bf16 precision

#### Output Configuration
- `--output_dir`: Root directory for outputs. Final path: `{output_dir}/siq-vl_{vision_backbone}_{text_backbone}_{stage}_{datetime}/{stage}` (default: `./checkpoints`)
- `--logging_steps`: Steps between logging (default: 10)
- `--save_steps`: Steps between checkpoints (default: 500)
- `--eval_steps`: Steps between evaluation (default: 100)
- `--max_eval_samples`: Maximum samples for evaluation (default: 2, set higher for meaningful eval)
- `--gen_steps`: Steps between generation evaluation (default: 100)
- `--gen_samples`: Number of fixed samples for generation evaluation (default: 20)
- `--gen_max_new_tokens`: Maximum new tokens for generation (default: 128)
- `--gen_temperature`: Temperature for generation (default: 0.0)
- `--gen_num_beams`: Number of beams for generation (default: 1)

#### Distributed Training
- `--use_distributed`: Enable distributed training (auto-detected if multiple GPUs available)
- `--no_distributed`: Disable distributed training

#### Hugging Face Hub
- `--push_to_hub`: Push final checkpoint to Hugging Face Hub (default: False)
- `--hub_model_id`: Optional explicit Hub model ID (default: auto-generated from model names and stage)

### Distributed Training

For multi-GPU training, use Accelerate:

```bash
accelerate launch \
    --dispatch_batches=false \
    --split_batches=false \
    scripts/train.py \
    --freeze_text_model \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    ...
```

### Publishing Checkpoints to the Hugging Face Hub

You can optionally publish trained checkpoints to the Hugging Face Hub so others can use the models without retraining.

- **Naming convention**: Repos are named as  
    `siq-vl_{vision_backbone}_{text_backbone}_{stage}`  
    For example: `siq-vl_siglip2-base-patch16-224_qwen2.5-0.5b-instruct_stage1`.

- **Stage inference**: The stage suffix (e.g., `stage1`, `stage2`) is automatically inferred from your `--project` name and/or `--output_dir`.  
  - Stage 1 runs launched via `scripts/train_stage_1.sh` will typically publish as `..._stage1`.  
  - Stage 2 runs launched via `scripts/train_stage_2.sh` will typically publish as `..._stage2`.

- **W&B integration**:
  - The Hub commit message includes the W&B run URL (when available).
  - A lightweight Hub git tag of the form `wandb-{run_id}` is created, whose message contains the W&B run URL.

#### Example: Publish Stage 1 Model (MacBook quick run)

```bash
bash scripts/train_stage_1.sh \
  --push_to_hub
```

This will:

- Train Stage 1 using the MacBook defaults.
- Save the final model under `./checkpoints/siq-vl_{vision}_{text}_{stage}_{datetime}/stage1`.
- Create (or reuse) a Hub repo named like:
  - `siq-vl_siglip2-base-patch16-224_qwen2.5-0.5b-instruct_stage1`
- Upload all files from the final checkpoint directory.
- Add a Hub tag `wandb-{run_id}` with a message that includes the W&B run URL.

#### Example: Publish Stage 2 Model (AWS p4d full run)

```bash
STAGE=2 bash scripts/train_launch.sh \
  --push_to_hub
```

This will:

- Train Stage 2 (full finetuning) using the AWS p4d defaults.
- Save the final model under `./checkpoints/siq-vl_{vision}_{text}_{stage}_{datetime}/stage2`.
- Create (or reuse) a Hub repo named like:
  - `siq-vl_siglip2-large-patch16-512_qwen2.5-1.5b-instruct_stage2`
- Upload all files from the final checkpoint directory.
- Add a Hub tag `wandb-{run_id}` with a message that includes the W&B run URL.

> To override the default repo id (for example to push under an organization), pass:
> `--hub_model_id your-org/siq-vl_siglip2-base-patch16-224_qwen2.5-0.5b-instruct_stage1`.

## Project Structure

```
SiQ_VL/
├── siq_vl/                     # Main package
│   ├── model/
│   │   └── modeling.py         # SiQ_VLForCausalLM architecture
│   ├── kernels/
│   │   └── fused_linear_ce.py  # Fused Linear-CE with grad-in-forward
│   ├── processing.py           # SiQ_VLProcessor for multimodal inputs
│   ├── dataset.py              # VQADataset for efficient data loading
│   ├── collator.py             # Data collator + PackingCollator
│   └── callbacks.py            # Training callbacks (metrics, GPU cleanup)
├── scripts/
│   ├── train.py                # Main training script (Stage 1 & Stage 2)
│   ├── train_stage_1.sh        # Convenience script for Stage 1
│   ├── train_stage_2.sh        # Convenience script for Stage 2
│   ├── benchmark_real_efficiency.py  # Ground-truth efficiency measurement
│   ├── benchmark_batch_scaling.py    # Batch size sweep benchmark
│   ├── benchmark_cutile_e2e.py       # TileGym E2E benchmark
│   └── benchmark_flashoptim.py       # FlashOptim memory benchmark
├── docs/
│   ├── report/
│   │   ├── main.tex            # Technical report (LaTeX, arXiv-ready)
│   │   └── main.pdf            # Compiled report (21 pages)
│   ├── training_efficiency_plan.md  # Raw iteration log
│   └── traces/                 # Per-iteration measurement JSONs
├── checkpoints/                # Saved model checkpoints
└── lmms-eval/                  # Evaluation framework (optional)
```

## Development Roadmap

- [x] **Stage 1**: Projector alignment training
- [x] **Stage 2**: Language model fine-tuning with LoRA support
- [x] **Training Efficiency**: Single-GPU optimization (22.9x speedup, technical report)
- [x] **TileGym Integration**: Blackwell-native cuTile kernels (FA4, RMSNorm, SwiGLU, RoPE, fused CE)
- [x] **Sequence Packing**: FlexAttention-based packing with block-diagonal masks
- [ ] **Stage 3**: Supervised fine-tuning with chain-of-thought reasoning
- [ ] **Stage 4**: Reinforcement learning-based training (RLHF/DPO)
- [ ] **Distributed Training**: Multi-GPU scaling with FSDP/Modal
- [ ] Evaluation scripts and benchmark integration
- [ ] Model inference and deployment utilities

## Model Specifications

### Vision Encoder Specifications

- **Model Architecture**: SigLIP (SigLIP 2 SO400M or base model variants)
- **Training Status**: Parameters remain frozen throughout all training stages
- **Output Characteristics**: Produces vision features with configurable patch size and image resolution settings

### Projection Module Specifications

- **Architecture Type**: MLP (Multi-Layer Perceptron) with pixel shuffle operation
- **Functional Role**: Transforms vision encoder hidden dimensions to match language model embedding dimensions
- **Compression Mechanism**: Pixel shuffle operation reduces sequence length (e.g., 729 tokens → 182 tokens for 384×384 pixel images with shuffle factor of 2)
- **Default Pixel Shuffle Factor**: 2 (configurable via `--pixel_shuffle_factor`)
- **Architecture**: Pixel Shuffle → Linear Projection (no normalization layer)

### Language Model Specifications

- **Model Architecture**: Qwen2.5 (available in 0.5B, 1.5B, and larger parameter variants)
- **Training Status**: 
  - **Stage 1**: Parameters remain frozen; only projection module is trained
  - **Stage 2 and subsequent stages**: Parameters are unfrozen for fine-tuning (supports LoRA or full fine-tuning)
- **LoRA Support**: Stage 2 supports optional LoRA fine-tuning for efficient training (recommended)
- **Special Token Handling**: Utilizes Qwen's native special tokens including `<|image_pad|>`, `<|vision_start|>`, and `<|vision_end|>`

## Usage Examples

### Loading a Stage 1 Checkpoint

The following code demonstrates how to load a trained Stage 1 checkpoint for inference:

```python
from siq_vl.model import SiQ_VLModel
from siq_vl.processing import SiQ_VLProcessor
from transformers import AutoImageProcessor, AutoTokenizer
from PIL import Image
import torch
import json
import os

# Load checkpoint configuration
checkpoint_dir = "./checkpoints/siq_vlm_stage1"
with open(os.path.join(checkpoint_dir, "model_config.json"), "r") as f:
    model_config = json.load(f)

# Load processor (saved with the model)
processor = SiQ_VLProcessor.from_pretrained(checkpoint_dir)

# Initialize model with saved configuration
model = SiQ_VLModel(
    vision_model_path=model_config["vision_model_path"],
    llm_model_path=model_config["llm_model_path"],
    freeze_llm=True  # Stage 1 uses frozen LLM
)

# Load the trained weights
model.load_state_dict(torch.load(
    os.path.join(checkpoint_dir, "pytorch_model.bin"),
    map_location="cpu"
))
model.eval()

# Prepare inputs
image = Image.open("path/to/image.jpg")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# Process and forward
inputs = processor(text=messages, images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Generate response (example)
# Note: Full generation code depends on your inference setup
```

### Initializing Model from Base Architectures

The following example demonstrates model initialization from pre-trained base models for Stage 1 training:

```python
model = SiQ_VLModel(
    vision_model_path="google/siglip-so400m-patch14-384",
    llm_model_path="Qwen/Qwen2.5-0.5B-Instruct",
    freeze_llm=True  # Stage 1: freeze LLM
)
```

## Training Notes and Recommendations

### Stage 1 Training Considerations

- **Memory Requirements**: Training requires substantial VRAM. For GPUs with 24GB VRAM, recommended batch sizes range from 4-8 with gradient accumulation enabled.
- **Numerical Precision**: Qwen models exhibit optimal performance with bfloat16 precision. The use of float16 precision is not recommended for Qwen architectures.
- **Overfitting Behavior**: Vision-language models may exhibit rapid overfitting. Approximately 1000 training steps typically suffice for projector alignment in Stage 1.
- **Checkpoint Format**: Models are saved in PyTorch format (`.bin` files) to circumvent potential safetensors compatibility issues.
- **Learning Rate Selection**: Stage 1 employs a learning rate of 1e-3 for projector alignment. Subsequent stages utilize lower learning rates (1e-5 to 2e-5) for language model fine-tuning.

### Multi-Stage Training Considerations

- **Progressive Checkpoint Loading**: Each training stage builds upon checkpoints from previous stages. Stage 1 checkpoints must be loaded prior to initiating Stage 2 training.
- **Parameter Freezing Strategy**: 
  - Stage 1: Vision encoder and language model parameters remain frozen
  - Stage 2 and subsequent stages: Only vision encoder parameters remain frozen
- **Dataset Progression**: Training stages employ increasingly specialized datasets designed to target specific model capabilities.

## Contributing

Contributions to this project are welcome. Please submit pull requests for review.

## License

This project is licensed under the MIT License:

```
MIT License

Copyright (c) 2025 SiQ-VL Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

This work builds upon the following open-source contributions:

- **SigLIP2** (Zhai et al., 2023): Vision encoder architecture [[arXiv](https://arxiv.org/abs/2502.14786)]
- **Qwen2.5** (Qwen Team, 2024): Language model architecture [[arXiv](https://arxiv.org/abs/2412.15115)]
- **HuggingFace Transformers** (Wolf et al., 2020): Deep learning framework [[GitHub](https://github.com/huggingface/transformers)]
- **FineVision Dataset** (HuggingFace, 2025): Open dataset for data-centric VLM training [[HuggingFace](https://huggingface.co/datasets/HuggingFaceM4/FineVision)]
- **TileGym** (NVIDIA, 2025): Production cuTile kernels for Blackwell GPUs (FA4, fused ops)
- **Liger-Kernel** (LinkedIn, 2024): Triton-based fused kernels for LLM training [[GitHub](https://github.com/linkedin/Liger-Kernel)]
- **FlashAttention** (Dao et al., 2022): Memory-efficient attention [[GitHub](https://github.com/Dao-AILab/flash-attention)]
- **FlashOptim** (Databricks, 2025): Memory-efficient optimizer states [[GitHub](https://github.com/databricks/flashoptim)]

