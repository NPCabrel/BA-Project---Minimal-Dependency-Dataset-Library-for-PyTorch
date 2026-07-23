"""
Multi-processing DataLoader with staging queue and batch queue.
Uses multiprocessing for true parallelism (no GIL).
"""
import multiprocessing as mp
import threading
import time
import json
import os
from typing import Optional, Callable

import torch

from .sampler import LockFreeSampler
from .metrics import MetricsTracker


def _worker_fn(worker_id, dataset, indices, staging_queue, stop_event):
    """Worker process: loads samples and pushes to staging queue."""
    for idx in indices:
        if stop_event.is_set():
            break
        sample = dataset[idx]
        staging_queue.put(sample)


def _orchestrator_fn(staging_queue, batch_queue, batch_size, collate_fn, stop_event, result_queue):
    """Orchestrator process: pulls from staging, collates, pushes to batch."""
    buffer = []
    batches_produced = 0
    empty_events = 0
    full_events = 0
    wait_time = 0.0

    while not stop_event.is_set():
        try:
            t0 = time.perf_counter()
            sample = staging_queue.get(timeout=0.1)
            wait_time += time.perf_counter() - t0
            buffer.append(sample)
            if len(buffer) >= batch_size:
                batch_samples = buffer[:batch_size]
                buffer = buffer[batch_size:]
                batch = collate_fn(batch_samples)
                batch_queue.put(batch)
                batches_produced += 1
        except:
            empty_events += 1
            continue

    while len(buffer) >= batch_size:
        batch = collate_fn(buffer[:batch_size])
        buffer = buffer[batch_size:]
        batch_queue.put(batch)
        batches_produced += 1

    result_queue.put({
        "batches_produced": batches_produced,
        "empty_events": empty_events,
        "full_events": full_events,
        "wait_time": wait_time,
    })


class DataLoaderMP:
    """
    Multi-processing DataLoader (no GIL limitation).

    Args:
        dataset: Dataset instance (must be picklable)
        batch_size: Samples per batch
        num_workers: Number of worker processes
        max_staging_size: Max samples in staging queue
        max_batch_queue_size: Max batches in output queue
        collate_fn: Custom collate function
        metrics_dir: Directory for metrics JSON
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

        self.collate_fn = collate_fn or self._default_collate

        self._ctx = mp.get_context('fork')
        self.staging_queue = self._ctx.Queue(maxsize=max_staging_size)
        self.batch_queue = self._ctx.Queue(maxsize=max_batch_queue_size)
        self.result_queue = self._ctx.Queue()

        self.sampler = LockFreeSampler(len(dataset), num_workers, shuffle=True)

        self._stop_event = self._ctx.Event()
        self._workers = []
        self._orchestrator = None

        self._batch_count = 0

    def _default_collate(self, samples):
        images = torch.stack([s[0] for s in samples])
        labels = [s[1] for s in samples]
        if isinstance(labels[0], str):
            labels = torch.tensor([int(l.replace('n', '')) for l in labels])
        else:
            labels = torch.tensor(labels)
        return images, labels

    def __iter__(self):
        self._stop_event.clear()

        # Start workers
        self._workers = []
        for w in range(self.num_workers):
            indices = self.sampler.get_partition(w)
            p = self._ctx.Process(
                target=_worker_fn,
                args=(w, self.dataset, indices, self.staging_queue, self._stop_event)
            )
            p.start()
            self._workers.append(p)

        # Start orchestrator
        self._orchestrator = self._ctx.Process(
            target=_orchestrator_fn,
            args=(self.staging_queue, self.batch_queue, self.batch_size,
                  self.collate_fn, self._stop_event, self.result_queue)
        )
        self._orchestrator.start()

        return self

    def __next__(self):
        all_done = all(not p.is_alive() for p in self._workers + [self._orchestrator])
        if self.batch_queue.empty() and all_done:
            self._cleanup()
            raise StopIteration
        try:
            return self.batch_queue.get(timeout=1.0)
        except:
            if self.batch_queue.empty() and all(not p.is_alive() for p in self._workers + [self._orchestrator]):
                self._cleanup()
                raise StopIteration
            return self.batch_queue.get()

    def _cleanup(self):
        self._stop_event.set()
        for p in self._workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
        if self._orchestrator and self._orchestrator.is_alive():
            self._orchestrator.terminate()
            self._orchestrator.join(timeout=2.0)

        # Get orchestrator stats if available
        try:
            stats = self.result_queue.get_nowait()
        except:
            stats = {"batches_produced": 0, "empty_events": 0, "full_events": 0, "wait_time": 0}

        if self.metrics_dir:
            os.makedirs(self.metrics_dir, exist_ok=True)
            path = os.path.join(self.metrics_dir, f"metrics_mp_w{self.num_workers}_bs{self.batch_size}.json")
            with open(path, 'w') as f:
                json.dump(stats, f, indent=2)

    def set_epoch(self, seed: int = None):
        self.sampler.reshuffle(seed)
