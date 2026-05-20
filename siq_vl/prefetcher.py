"""
CUDA stream-based data prefetcher for overlapping H2D transfers with compute.

The default PyTorch DataLoader with pin_memory=True still performs synchronous
.to(device) on the default CUDA stream, blocking compute until the transfer
completes. This prefetcher uses a dedicated CUDA stream to pipeline:

  Step N compute (default stream)  |  Step N+1 H2D transfer (prefetch stream)
  ─────────────────────────────────|──────────────────────────────────────────
  forward + backward               |  next_batch.to(device, non_blocking=True)

This hides the H2D latency entirely when compute time > transfer time.
"""

import torch
from torch.utils.data import DataLoader


class CUDAPrefetcher:
    """
    Wraps a DataLoader to prefetch the next batch on a separate CUDA stream.

    Usage:
        prefetcher = CUDAPrefetcher(dataloader, device=torch.device("cuda"))
        for batch in prefetcher:
            # batch is already on GPU, ready for compute
            outputs = model(**batch)
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._iter = None
        self._next_batch = None

    def __iter__(self):
        self._iter = iter(self.dataloader)
        self._preload()
        return self

    def _to_device(self, data):
        """Recursively move tensors to device with non_blocking=True."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._to_device(v) for v in data)
        return data

    def _preload(self):
        """Load next batch onto GPU using the prefetch stream."""
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self._next_batch = self._to_device(batch)

    def __next__(self):
        # Wait for the prefetch stream to finish the current batch transfer
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

        if self._next_batch is None:
            raise StopIteration

        batch = self._next_batch

        # Ensure tensors record the dependency on the prefetch stream
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    v.record_stream(torch.cuda.current_stream(self.device))
        elif isinstance(batch, (list, tuple)):
            for v in batch:
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    v.record_stream(torch.cuda.current_stream(self.device))

        # Start prefetching the NEXT batch while this one is being consumed
        self._preload()

        return batch

    def __len__(self):
        return len(self.dataloader)

    def reset(self):
        """Reset the iterator (for multi-epoch training)."""
        self._iter = iter(self.dataloader)
        self._preload()
