import os
from typing import Iterator, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info


def _to_pil_rgb(image: Union[str, Image.Image]) -> Image.Image:
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
        return image

    # Case 2 — It is a file path (string)
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
        return img

    # Case 3 — torch.Tensor or numpy array
    # Convert to numpy array for unified processing
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.array(image)

    # Ensure the array has at least 3 dimensions
    # Typical formats are:
    # - H x W x C  (channels_last)
    # - C x H x W  (channels_first)
    if arr.ndim == 3:
        # If it's channels-first (C, H, W) → convert to (H, W, C)
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

    # If the array is grayscale (H, W) → duplicate into RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # If the last dimension is single-channel, replicate it 3 times
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # If the last dimension is 4 (RGBA), convert to RGB
    if arr.ndim == 3 and arr.shape[-1] == 4:
        # Remove alpha channel
        arr = arr[:, :, :3]

    # Ensure values are uint8 (required by PIL)
    if arr.dtype != np.uint8:
        # Clip out-of-range values and cast
        arr = np.clip(arr, 0, 255).astype("uint8")

    # Final safety check: ensure we have exactly 3 channels
    if arr.ndim != 3 or arr.shape[-1] != 3:
        # If still not RGB, force convert through PIL
        temp_img = Image.fromarray(arr) if arr.ndim >= 2 else Image.new("RGB", (1, 1))
        return temp_img.convert("RGB")

    # Convert final array into a proper RGB PIL image
    return Image.fromarray(arr, mode="RGB")


class VQAIterableDataset(IterableDataset):
    def __init__(
        self,
        hf_dataset,
        processor,
        *,
        return_raw_data=False,
        max_length=1024,
        cache_dir=None,
    ):
        """
        hf_dataset: HuggingFace dataset object
        processor: Qwen-VL or LLaVA processor
        max_length: model context
        cache_dir: store pixel_values + input_ids once per image/turn
        """
        self.dataset = hf_dataset
        self.processor = processor
        self.return_raw_data = return_raw_data
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.ignore_index = -100

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.pixel_cache = os.path.join(cache_dir, "pixels")
            self.text_cache = os.path.join(cache_dir, "text")
            os.makedirs(self.pixel_cache, exist_ok=True)
            os.makedirs(self.text_cache, exist_ok=True)

        # tokenizer ids
        # https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json
        tok = processor.tokenizer
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.assistant_id = tok.encode("assistant", add_special_tokens=False)[0]
        self.image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")

    def _load_or_process_image(self, idx, image):
        """pixel cache per image"""
        if not self.cache_dir:
            return self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        path = f"{self.pixel_cache}/{idx}.pt"
        if os.path.exists(path):
            return torch.load(path)

        pixel = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        torch.save(pixel.cpu(), path)
        return pixel

    def _load_or_process_turn(self, idx, t_idx, q, a):
        """token cache per text turn"""
        if not self.cache_dir:
            msgs = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": q}],
                },
                {"role": "assistant", "content": a},
            ]
            enc = self.processor(
                text=msgs,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            return enc.input_ids[0]

        path = f"{self.text_cache}/{idx}_{t_idx}.pt"
        if os.path.exists(path):
            return torch.load(path)

        msgs = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": q}],
            },
            {"role": "assistant", "content": a},
        ]
        enc = self.processor(
            text=msgs, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        input_ids = enc.input_ids[0]
        torch.save(input_ids.cpu(), path)
        return input_ids

    def _mask_labels(self, input_ids):
        """
        Mask all tokens except the assistant's response.

        Sequence structure:
        <|im_start|>assistant\nThe image depicts...<|im_end|>

        We want to predict only: "The image depicts...", not the formatting tokens.
        """
        labels = torch.ones_like(input_ids) * self.ignore_index
        ids = input_ids.tolist()

        try:
            for i, x in enumerate(ids):
                # Find: <|im_start|> followed by assistant token
                if (
                    x == self.im_start
                    and i + 1 < len(ids)
                    and ids[i + 1] == self.assistant_id
                ):
                    # Position breakdown:
                    # i     : <|im_start|> (151644)
                    # i+1   : assistant (77091)
                    # i+2   : \n (198) ← Skip this!
                    # i+3   : Start of actual answer

                    # Start labeling from i+3 (after <|im_start|>, assistant, and \n)
                    s = i + 3

                    # Find the end marker
                    try:
                        e = ids.index(self.im_end, s)
                    except ValueError:
                        e = len(ids)

                    # Set labels for the actual answer content
                    labels[s:e] = input_ids[s:e]
                    break
        except Exception as ex:
            # If something goes wrong, keep all labels as ignore_index
            print(f"Warning: Failed to mask labels: {ex}")
            pass

        # Also mask image tokens (they should never be predicted)
        labels[input_ids == self.image_token_id] = self.ignore_index
        return labels

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
            pixel = self._load_or_process_image(i, image)

            for t_idx, turn in enumerate(item["texts"]):
                q = turn["user"]
                a = turn["assistant"]

                input_ids = self._load_or_process_turn(i, t_idx, q, a)
                input_ids = input_ids.clone()
                labels = self._mask_labels(input_ids)
                out = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "pixel_values": pixel,
                }
                if self.return_raw_data:
                    yield {
                        **out,
                        "question": q,
                        "answer": a,
                    }
                else:
                    yield out
