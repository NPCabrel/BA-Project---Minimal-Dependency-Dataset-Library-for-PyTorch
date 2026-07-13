"""
Multi-threaded DataLoader with staging queue and batch queue.

Architecture:
    Main thread ← batch_queue ← Collator ← staging_queue ← Workers ← Sampler
"""
import threading
import time
import json
import os
from typing import Optional, Callable

import torch

from .sampler import LockFreeSampler
from .monitored_queue import MonitoredQueue
from .metrics import MetricsTracker


class DataLoader:
    """
    Multi-threaded DataLoader with instrumentation.

    Args:
        dataset: BaseDataset instance
        batch_size: Samples per batch
        num_workers: Number of loading threads
        max_staging_size: Max samples in staging queue
        max_batch_queue_size: Max batches in output queue
        collate_fn: Custom collate function
        metrics_dir: Directory to save metrics JSON (None = no saving)
    """

    def __init__(self, dataset, batch_size: int,
                 num_workers: int = 1,
                 max_staging_size: int = 256,
                 max_batch_queue_size: int = 8,
                 collate_fn: Optional[Callable] = None,
                 metrics_dir: Optional[str] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metrics_dir = metrics_dir

        # Use monitored queues for instrumentation
        self.staging_queue = MonitoredQueue(maxsize=max_staging_size, name="staging")
        self.batch_queue = MonitoredQueue(maxsize=max_batch_queue_size, name="batch")

        self.collate_fn = collate_fn or self._default_collate

        self.sampler = LockFreeSampler(
            len(dataset), num_workers, shuffle=True
        )

        self._stop_event = threading.Event()
        self._threads = []
        self._tracker = MetricsTracker(num_workers)

    def _worker(self, worker_id: int):
        """Load samples and push to staging queue."""
        indices = self.sampler.get_partition(worker_id)
        wm = self._tracker.get_worker(worker_id)

        for idx in indices:
            if self._stop_event.is_set():
                break

            # Time the loading + preprocessing
            wm.start_sample()
            sample = self.dataset[idx]
            wm.end_sample()

            # Put into staging queue (may block if full)
            put_start = time.perf_counter()
            self.staging_queue.put(sample)
            wm.record_idle(time.perf_counter() - put_start)

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
                    self._tracker.record_batch()
            except Exception:
                # queue.Empty or other — continue polling
                continue

        # Flush remaining
        while len(buffer) >= self.batch_size:
            batch = self.collate_fn(buffer[:self.batch_size])
            buffer = buffer[self.batch_size:]
            self.batch_queue.put(batch)
            self._tracker.record_batch()

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
        self._tracker = MetricsTracker(self.num_workers)

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
        """Stop threads and save metrics."""
        self._stop_event.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)

        # Save metrics
        summary = self._tracker.summary(
            self.staging_queue.stats(),
            self.batch_queue.stats()
        )

        if self.metrics_dir:
            os.makedirs(self.metrics_dir, exist_ok=True)
            path = os.path.join(
                self.metrics_dir,
                f"metrics_w{self.num_workers}_bs{self.batch_size}.json"
            )
            with open(path, 'w') as f:
                json.dump(summary, f, indent=2)

        return summary

    def set_epoch(self, seed: int = None):
        """Reshuffle sampler and reset metrics for new epoch."""
        self.sampler.reshuffle(seed)
        self.staging_queue.reset()
        self.batch_queue.reset()
        self._tracker.reset()
