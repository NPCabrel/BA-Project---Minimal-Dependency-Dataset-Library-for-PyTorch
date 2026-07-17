"""
Per-worker metrics collection. Merged at epoch end for pipeline summary.
"""
import time
import threading


class WorkerMetrics:
    """Tracks timing for a single worker."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.samples_processed = 0
        self.busy_time = 0.0     # Time spent loading + preprocessing
        self.idle_time = 0.0     # Time spent waiting (queue full, etc.)
        self._current_start = None

    def start_sample(self):
        self._current_start = time.perf_counter()

    def end_sample(self):
        if self._current_start:
            self.busy_time += time.perf_counter() - self._current_start
            self.samples_processed += 1
            self._current_start = None

    def record_idle(self, duration: float):
        self.idle_time += duration

    def snapshot(self):
        return {
            "worker_id": self.worker_id,
            "samples_processed": self.samples_processed,
            "busy_time_s": round(self.busy_time, 4),
            "idle_time_s": round(self.idle_time, 4),
            "utilization_pct": round(
                100 * self.busy_time / (self.busy_time + self.idle_time + 1e-8), 2
            ),
        }


class MetricsTracker:
    """
    Collects and merges metrics from workers and queues.

    Usage:
        tracker = MetricsTracker(num_workers=4)
        # Workers call tracker.get_worker_metrics(wid) to record
        # At epoch end: tracker.summary(queue_stats) -> dict
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.workers = [WorkerMetrics(i) for i in range(num_workers)]
        self.epoch_start = time.perf_counter()
        self.batches_produced = 0
        self._lock = threading.Lock()

    def get_worker(self, worker_id: int) -> WorkerMetrics:
        return self.workers[worker_id]

    def record_batch(self):
        with self._lock:
            self.batches_produced += 1

    def summary(self, staging_stats: dict, batch_stats: dict):
        """Merge all metrics and return a dict."""
        elapsed = time.perf_counter() - self.epoch_start

        worker_summaries = [w.snapshot() for w in self.workers]
        total_busy = sum(w["busy_time_s"] for w in worker_summaries)
        total_idle = sum(w["idle_time_s"] for w in worker_summaries)

        return {
            "epoch_time_s": round(elapsed, 2),
            "num_workers": self.num_workers,
            "batches_produced": self.batches_produced,
            "workers": worker_summaries,
            "aggregate": {
                "total_busy_time_s": round(total_busy, 4),
                "total_idle_time_s": round(total_idle, 4),
                "avg_utilization_pct": round(
                    100 * total_busy / (total_busy + total_idle + 1e-8), 2
                ),
            },
            "staging_queue": staging_stats,
            "batch_queue": batch_stats,
        }

    def reset(self):
        """Reset for new epoch."""
        self.workers = [WorkerMetrics(i) for i in range(self.num_workers)]
        self.epoch_start = time.perf_counter()
        self.batches_produced = 0
