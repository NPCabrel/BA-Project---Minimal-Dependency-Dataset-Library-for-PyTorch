"""
Multi-threaded DataLoader with staging queue and batch queue.

Architecture:
    Main thread ← batch_queue ← Collator ← staging_queue ← Workers ← Sampler
"""
import threading
import queue
from typing import Optional, Callable

import torch

from .sampler import LockFreeSampler


class DataLoader:
    """
    Multi-threaded DataLoader.

    Args:
        dataset: BaseDataset instance
        batch_size: Samples per batch
        num_workers: Number of loading threads
        max_staging_size: Max samples in staging queue
        max_batch_queue_size: Max batches in output queue
        collate_fn: Custom collate function
    """

    def __init__(self, dataset, batch_size: int,
                 num_workers: int = 1,
                 max_staging_size: int = 256,
                 max_batch_queue_size: int = 8,
                 collate_fn: Optional[Callable] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.staging_queue = queue.Queue(maxsize=max_staging_size)
        self.batch_queue = queue.Queue(maxsize=max_batch_queue_size)

        self.collate_fn = collate_fn or self._default_collate

        self.sampler = LockFreeSampler(
            len(dataset), num_workers, shuffle=True
        )

        self._stop_event = threading.Event()
        self._threads = []

    def _worker(self, worker_id: int):
        """Load samples and push to staging queue."""
        indices = self.sampler.get_partition(worker_id)
        for idx in indices:
            if self._stop_event.is_set():
                break
            sample = self.dataset[idx]
            self.staging_queue.put(sample)

    def _orchestrator(self):
        """Pull from staging queue, collate when batch ready."""
        buffer = []
        while not self._stop_event.is_set():
            try:
                sample = self.staging_queue.get(timeout=0.1)
                buffer.append(sample)
                if len(buffer) >= self.batch_size:
                    batch_samples = buffer[:self.batch_size]
                    buffer = buffer[self.batch_size:]
                    batch = self.collate_fn(batch_samples)
                    self.batch_queue.put(batch)
            except queue.Empty:
                continue
        # Flush remaining
        while len(buffer) >= self.batch_size:
            batch = self.collate_fn(buffer[:self.batch_size])
            buffer = buffer[self.batch_size:]
            self.batch_queue.put(batch)

    def _default_collate(self, samples):
        """Stack tensors into a batch."""
        images = torch.stack([s[0] for s in samples])
        labels = [s[1] for s in samples]
        if isinstance(labels[0], str):
            labels = torch.tensor([int(l.replace('n', '')) for l in labels])
        else:
            labels = torch.tensor(labels)
        return images, labels

    def __iter__(self):
        self._stop_event.clear()
        self._threads = []

        for w in range(self.num_workers):
            t = threading.Thread(target=self._worker, args=(w,))
            t.start()
            self._threads.append(t)

        orch = threading.Thread(target=self._orchestrator)
        orch.start()
        self._threads.append(orch)

        return self

    def __next__(self):
        all_done = all(not t.is_alive() for t in self._threads)
        if self.batch_queue.empty() and all_done:
            self._cleanup()
            raise StopIteration
        return self.batch_queue.get()

    def _cleanup(self):
        """Stop all threads."""
        self._stop_event.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)

    def set_epoch(self, seed: int = None):
        """Reshuffle sampler for new epoch."""
        self.sampler.reshuffle(seed)
