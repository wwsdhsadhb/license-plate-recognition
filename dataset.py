"""
Dataset utilities for license plate recognition training.
Supports CCPD (Chinese City Parking Dataset) and custom datasets.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from utils.ctc_decoder import CHAR2IDX, CHARS


class LicensePlateDataset(Dataset):
    """
    Dataset for CRNN training.
    Expected directory structure:
        data_dir/
            images/   *.jpg
            labels/   *.txt  (one plate string per file, same stem as image)

    Or flat: image filename encodes the label (CCPD-style).
    """

    def __init__(
        self,
        data_dir: str,
        img_h: int = 32,
        img_w: int = 100,
        augment: bool = True,
        max_label_len: int = 8,
    ):
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        self.max_label_len = max_label_len

        img_dir = os.path.join(data_dir, "images")
        lbl_dir = os.path.join(data_dir, "labels")

        self.samples: List[Tuple[str, str]] = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    label = f.read().strip()
                if label:
                    self.samples.append((img_path, label))

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        image = cv2.resize(image, (self.img_w, self.img_h))

        if self.augment:
            image = self._augment(image)

        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Encode label
        encoded = [CHAR2IDX.get(c, 0) for c in label]
        return tensor, torch.tensor(encoded, dtype=torch.long), len(encoded)

    @staticmethod
    def _augment(image: np.ndarray) -> np.ndarray:
        """Light augmentation: brightness jitter + random blur."""
        # Brightness
        factor = np.random.uniform(0.7, 1.3)
        image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        # Optional blur
        if np.random.rand() < 0.2:
            ksize = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return image


def collate_fn(batch):
    """Custom collate to handle variable-length label sequences."""
    images, labels, label_lens = zip(*batch)
    images = torch.stack(images, dim=0)
    # Pad labels
    max_len = max(label_lens)
    padded = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, lbl in enumerate(labels):
        padded[i, : len(lbl)] = lbl
    return images, padded, torch.tensor(label_lens, dtype=torch.long)


def build_dataloader(
    data_dir: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True,
) -> DataLoader:
    dataset = LicensePlateDataset(data_dir, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
