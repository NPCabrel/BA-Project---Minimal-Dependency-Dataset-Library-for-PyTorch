"""
Parquet-backed Dataset. Reads image bytes and labels from a Parquet file.
"""
import io
from typing import Optional, Callable

import torch
from PIL import Image
import torchvision.transforms as T
import pyarrow.parquet as pq


class ParquetDataset:
    """
    Dataset that reads samples from a Parquet file.

    Parquet schema:
        - image: bytes (JPEG/PNG)
        - label: string or int

    Args:
        parquet_path: Path to .parquet file
        max_samples: Limit number of samples (for testing)
        transform: Optional torchvision transform (applied after preprocess)
    """

    def __init__(self, parquet_path: str, max_samples: int = None,
                 transform: Optional[Callable] = None):
        self.parquet_path = parquet_path
        self.transform = transform

        # Read entire table into memory
        self._table = pq.read_table(parquet_path)
        self._length = len(self._table)
        if max_samples:
            self._length = min(self._length, max_samples)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        # Read one row from the in-memory table
        row = self._table.slice(index, 1).to_pylist()[0]

        img_bytes = row['image']
        label = row['label']

        # Decode image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Preprocessing
        img, label = self.preprocess(img, label)

        if self.transform:
            img = self.transform(img)

        return img, label

    def preprocess(self, img: Image.Image, label):
        """Override for augmentation. Default: resize to 64x64, convert to tensor."""
        img = img.resize((64, 64))
        return T.ToTensor()(img), label
