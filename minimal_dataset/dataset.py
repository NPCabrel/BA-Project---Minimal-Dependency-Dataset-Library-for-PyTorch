"""
BaseDataset class for reading image samples.
Users extend with preprocess() for augmentation.
"""
import os
from typing import Optional, Callable

import torch
from PIL import Image
import torchvision.transforms as T


class BaseDataset:
    def __init__(self, data_dir: str, max_samples: int = None,
                 transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        for class_dir in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                for fname in sorted(os.listdir(class_path)):
                    if fname.endswith(('.JPEG', '.jpg', '.png')):
                        self.samples.append((os.path.join(class_path, fname), class_dir))
                        if max_samples and len(self.samples) >= max_samples:
                            break
            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        img, label = self.preprocess(img, label)
        if self.transform:
            img = self.transform(img)
        return img, label

    def preprocess(self, img: Image.Image, label):
        img = img.resize((64, 64))
        return T.ToTensor()(img), label
