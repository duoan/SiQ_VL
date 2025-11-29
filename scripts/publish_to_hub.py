#!/usr/bin/env python3
"""
Publish an already trained SiQ-VL checkpoint to the Hugging Face Hub.

This is useful when you have finished training and just want to push an
existing checkpoint directory without re-running training.

Naming convention for the Hub repo:
    siq-vl_{vision_backbone}_{llm_backbone}_{stage}
Example:
    siq-vl_siglip2-base-patch16-224_qwen2.5-0.5b-instruct_stage1
"""

import argparse
import os
import shutil
import re
from typing import Optional

from huggingface_hub import HfApi


def infer_stage_name_from_path(path: str) -> str:
    """
    Infer stage name like 'stage1' / 'stage2' from a checkpoint path.
    Looks for 'stage1', 'stage_1', 'stage-1', etc. in any path component.
    """

    def _infer(s: str) -> Optional[str]:
        s = s.lower()
        m = re.search(r"stage[_\\s-]?(\\d+)", s)
        if m:
            return f"stage{m.group(1)}"
        return None

    parts = os.path.abspath(path).split(os.sep)
    for part in parts:
        stage = _infer(part)
        if stage is not None:
            return stage

    return "stage1"


def publish_to_hub(
    checkpoint_dir: str,
    hub_model_id: Optional[str] = None,
    wandb_run_url: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
) -> None:
    """
    Core publishing logic that can be reused from scripts or other modules.

    Args:
        checkpoint_dir: Path to the backbone checkpoint directory.
        hub_model_id: Optional explicit Hub repo id. If None, a default id
            following the siq-vl_{vision}_{llm}_{stage} convention is used.
        wandb_run_url: Optional W&B run URL to include in commit message/tag.
        wandb_run_id: Optional W&B run id for 'wandb-{run_id}' tag. If None,
            and wandb_run_url is provided, the id is guessed from the URL.
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    # Walk up the directory tree until we find something that looks like a
    # backbone folder:
    #
    #   ./checkpoints/siq-vl_siglip2-..._qwen.../stage1/final
    #   ./checkpoints/siq-vl_siglip2-..._qwen.../stage1/checkpoint-20
    #   ./checkpoints/siq_vlm_stage1/siglip2-...__qwen...
    #
    # We want:
    #   backbone_dir = "siq-vl_siglip2-..._qwen..."    (new)
    #   backbone_dir = "siglip2-...__qwen..."         (legacy)
    current = checkpoint_dir.rstrip(os.sep)
    backbone_dir = os.path.basename(current)

    while True:
        name = os.path.basename(current)
        parent = os.path.dirname(current)

        # New layout: starts with "siq-vl_..."
        if name.startswith("siq-vl_"):
            backbone_dir = name
            break

        # Legacy layout: contains "__" between vision and llm
        if "__" in name:
            backbone_dir = name
            break

        # If we've reached the top or there is no parent change, stop.
        if parent == current or not parent:
            # Fallback: keep whatever we had as backbone_dir (may be imperfect)
            break

        current = parent

    # Infer stage from the full path (e.g. '.../stage1', 'siq_vlm_stage1', etc.)
    stage_name = infer_stage_name_from_path(checkpoint_dir)

    # ------------------------------------------------------------------
    # Decode backbone_dir into vision_name and llm_name.
    #
    # Supported layouts:
    #   1) New layout run root:  siq-vl_{vision_backbone}_{llm_backbone}
    #   2) Legacy layout:        {vision_backbone}__{llm_backbone}
    # ------------------------------------------------------------------
    vision_name: str
    llm_name: str

    # Case 1: new layout with "siq-vl_" prefix
    if backbone_dir.startswith("siq-vl_"):
        body = backbone_dir[len("siq-vl_") :]
        # Backward compat: if legacy "__" separator is present inside body.
        if "__" in body:
            vision_name, llm_name = body.split("__", 1)
        else:
            # Split at the last underscore into vision and llm
            # (since internal underscores are normalized to hyphens)
            if "_" in body:
                vision_name, llm_name = body.rsplit("_", 1)
            else:
                vision_name, llm_name = body, "unknown-llm"
    # Case 2: legacy layout: {vision}__{llm}
    elif "__" in backbone_dir:
        vision_name, llm_name = backbone_dir.split("__", 1)
    # Fallback: treat the whole name as vision backbone
    else:
        vision_name, llm_name = backbone_dir, "unknown-llm"

    # Default repo id (single underscores between logical segments),
    # with a "siq-vl" prefix:
    #   siq-vl_{vision_backbone}_{llm_backbone}_{stage}
    default_repo_id = f"siq-vl_{vision_name}_{llm_name}_{stage_name}"
    hub_model_id = hub_model_id or default_repo_id

    # If no namespace is provided, resolve to the current authenticated user.
    # This avoids 404s like "models/siq-vl_.../preupload/main".
    if "/" not in hub_model_id:
        try:
            api_tmp = HfApi()
            user_info = api_tmp.whoami()
            username = user_info.get("name") or user_info.get("username")
            if username:
                hub_model_id = f"{username}/{hub_model_id}"
        except Exception:  # pragma: no cover - defensive
            print(
                ">>> Warning: Could not resolve Hugging Face username automatically. "
                "If you see RepositoryNotFoundError, try passing --hub_model_id "
                "with an explicit 'username/...' or 'org/...' prefix."
            )

    print(f">>> checkpoint_dir: {checkpoint_dir}")
    print(f">>> backbone_dir:   {backbone_dir}")
    print(f">>> stage:          {stage_name}")
    print(f">>> Hub model id:   {hub_model_id}")

    api = HfApi()
    api.create_repo(repo_id=hub_model_id, repo_type="model", exist_ok=True)

    # Ensure a README is present in the checkpoint directory so that it
    # becomes the model card on the Hub. If there is already a README.md
    # alongside the checkpoint, we leave it as-is. Otherwise, we copy the
    # project-root README.md into this folder.
    dst_readme = os.path.join(checkpoint_dir, "README.md")
    if not os.path.exists(dst_readme):
        # Project root is assumed to be the parent of the directory
        # containing this script (i.e., SiQ_VL/).
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_readme = os.path.join(project_root, "README.md")
        if os.path.exists(src_readme):
            print(f">>> Copying project README.md into checkpoint dir: {dst_readme}")
            shutil.copy2(src_readme, dst_readme)
        else:
            print(
                ">>> Warning: No project README.md found at project root; "
                "model card on Hub may be empty."
            )

    commit_message = f"Add {stage_name} checkpoint for {backbone_dir}"
    if wandb_run_url:
        commit_message += f" | W&B run: {wandb_run_url}"

    print(">>> Uploading files to the Hugging Face Hub...")
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=hub_model_id,
        repo_type="model",
        commit_message=commit_message,
    )

    # Handle optional W&B tag
    run_id = wandb_run_id
    if run_id is None and wandb_run_url:
        # Try to recover run id from URL (last path component)
        try:
            run_id = wandb_run_url.rstrip("/").split("/")[-1]
        except Exception:
            run_id = None

    if run_id is not None:
        tag_name = f"wandb-{run_id}"
        tag_message = f"W&B run: {wandb_run_url}" if wandb_run_url else None
        try:
            api.create_tag(
                repo_id=hub_model_id,
                repo_type="model",
                tag=tag_name,
                revision="main",
                message=tag_message,
            )
            print(f">>> Created Hub tag '{tag_name}' with message '{tag_message}'")
        except Exception as e:  # pragma: no cover - defensive
            print(f">>> Warning: Failed to create Hub tag '{tag_name}': {e}")

    print(">>> Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Publish an existing SiQ-VL checkpoint to the Hugging Face Hub",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help=(
            "Path to the *backbone* checkpoint directory, e.g. "
            "'./checkpoints/siq-vl_siglip2-base-patch16-224_qwen2.5-0.5b-instruct/stage1'"
        ),
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help=(
            "Optional explicit Hub model id (e.g. 'org/siq-vl_...'). "
            "If not provided, a name of the form "
            "'siq-vl_{vision_backbone}_{llm_backbone}_{stage}' will be used."
        ),
    )
    parser.add_argument(
        "--wandb_run_url",
        type=str,
        default=None,
        help=(
            "Optional W&B run URL to include in the Hub commit message and tag "
            "(e.g. 'https://wandb.ai/your_proj/runs/xxxxxx')."
        ),
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help=(
            "Optional W&B run id to build the tag name 'wandb-{run_id}'. "
            "If omitted but --wandb_run_url is provided, the run id will be "
            "guessed from the URL."
        ),
    )

    args = parser.parse_args()

    publish_to_hub(
        checkpoint_dir=args.checkpoint_dir,
        hub_model_id=args.hub_model_id,
        wandb_run_url=args.wandb_run_url,
        wandb_run_id=args.wandb_run_id,
    )


if __name__ == "__main__":
    main()


