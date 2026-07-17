"""
Thread-safe queue wrapper with built-in metrics tracking.
"""
import time
import queue
import threading


class MonitoredQueue:
    """
    Wraps queue.Queue with counters for puts, gets, empty events, and full events.

    Args:
        maxsize: Maximum queue size (0 = unlimited)
        name: Human-readable name for logging (e.g., "staging", "batch")
    """

    def __init__(self, maxsize: int = 0, name: str = "queue"):
        self._queue = queue.Queue(maxsize=maxsize)
        self.name = name
        self.maxsize = maxsize

        # Counters (thread-safe via internal lock)
        self._lock = threading.Lock()
        self.put_count = 0
        self.get_count = 0
        self.empty_events = 0
        self.full_events = 0
        self.total_put_wait = 0.0   # Time spent blocked on put()
        self.total_get_wait = 0.0   # Time spent blocked on get()

    def put(self, item, timeout: float = None):
        """Put item with timing. Blocks if full."""
        start = time.perf_counter()
        if timeout:
            self._queue.put(item, timeout=timeout)
        else:
            self._queue.put(item)
        wait_time = time.perf_counter() - start

        with self._lock:
            self.put_count += 1
            self.total_put_wait += wait_time
            if self._queue.full():
                self.full_events += 1

    def get(self, timeout: float = None):
        """Get item with timing. Blocks if empty."""
        start = time.perf_counter()
        try:
            if timeout:
                item = self._queue.get(timeout=timeout)
            else:
                item = self._queue.get()
        except queue.Empty:
            with self._lock:
                self.empty_events += 1
            raise

        wait_time = time.perf_counter() - start
        with self._lock:
            self.get_count += 1
            self.total_get_wait += wait_time
        return item

    def qsize(self):
        return self._queue.qsize()

    def empty(self):
        return self._queue.empty()

    def full(self):
        return self._queue.full()

    def stats(self):
        """Return current metrics snapshot."""
        with self._lock:
            return {
                "name": self.name,
                "maxsize": self.maxsize,
                "current_size": self._queue.qsize(),
                "put_count": self.put_count,
                "get_count": self.get_count,
                "empty_events": self.empty_events,
                "full_events": self.full_events,
                "total_put_wait_s": round(self.total_put_wait, 4),
                "total_get_wait_s": round(self.total_get_wait, 4),
            }

    def reset(self):
        """Reset counters (call at epoch start)."""
        with self._lock:
            self.put_count = 0
            self.get_count = 0
            self.empty_events = 0
            self.full_events = 0
            self.total_put_wait = 0.0
            self.total_get_wait = 0.0
