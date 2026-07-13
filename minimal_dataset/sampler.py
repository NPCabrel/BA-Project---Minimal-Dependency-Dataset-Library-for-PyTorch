"""
Lock-Free Sampler for distributing indices across workers.
No locking needed — each worker owns a private partition.
"""


class LockFreeSampler:
    """
    Partitions shuffled indices across workers.
    Each worker receives a contiguous slice of the full index list.

    Args:
        num_samples: Total number of samples
        num_workers: Number of DataLoader workers
        shuffle: Whether to shuffle indices
        seed: Random seed for reproducibility
    """

    def __init__(self, num_samples: int, num_workers: int,
                 shuffle: bool = True, seed: int = 42):
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seed = seed
        self.worker_partitions = self._partition()

    def _partition(self):
        import random
        rng = random.Random(self.seed)
        indices = list(range(self.num_samples))
        if self.shuffle:
            rng.shuffle(indices)

        chunk_size = len(indices) // self.num_workers
        partitions = []
        for i in range(self.num_workers):
            start = i * chunk_size
            if i < self.num_workers - 1:
                end = start + chunk_size
            else:
                end = len(indices)
            partitions.append(indices[start:end])
        return partitions

    def get_partition(self, worker_id: int):
        """Return the list of indices assigned to this worker."""
        return self.worker_partitions[worker_id]

    def reshuffle(self, seed: int = None):
        """Reshuffle for a new epoch."""
        self.seed = seed if seed is not None else self.seed + 1
        self.worker_partitions = self._partition()
