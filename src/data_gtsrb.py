from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


NUM_CLASSES = 43

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class GTSRBPaths:
    root: str
    train_csv: str
    test_csv: str

    @classmethod
    def from_root(cls, root: str) -> "GTSRBPaths":
        return cls(
            root=root,
            train_csv=os.path.join(root, "Train.csv"),
            test_csv=os.path.join(root, "Test.csv"),
        )


class GTSRBDataset(Dataset):
    """GTSRB image dataset.

    Reads (path, class_id) pairs from in-memory arrays to keep
    splits (train/val) independent from the CSV layout on disk.
    """

    def __init__(self, root: str, paths: list[str], labels: np.ndarray, transform):
        self.root = root
        self.paths = paths
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        rel_path = self.paths[idx]
        full_path = os.path.join(self.root, rel_path)
        img = Image.open(full_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def build_transforms(image_size: int, augment: bool):
    """Compose transforms for GTSRB.

    Important: horizontal flip is NOT applied — sign semantics depend on
    orientation ("turn left" vs "turn right"). Similarly we keep rotations
    mild and avoid vertical flip.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if augment:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 8, image_size + 8)),
                transforms.RandomCrop(image_size),
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.08, 0.08),
                    scale=(0.9, 1.1),
                    shear=5,
                ),
                transforms.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


@dataclass
class GTSRBSplits:
    root: str
    train_paths: list[str]
    val_paths: list[str]
    test_paths: list[str]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def make_gtsrb_splits(
    root: str,
    val_size: float = 0.15,
    random_state: int = 42,
) -> GTSRBSplits:
    paths = GTSRBPaths.from_root(root)

    train_df = pd.read_csv(paths.train_csv)
    test_df = pd.read_csv(paths.test_csv)

    train_paths_all = train_df["Path"].tolist()
    y_train_all = train_df["ClassId"].values.astype(np.int64)

    train_paths, val_paths, y_train, y_val = train_test_split(
        train_paths_all,
        y_train_all,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_all,
    )

    test_paths = test_df["Path"].tolist()
    y_test = test_df["ClassId"].values.astype(np.int64)

    return GTSRBSplits(
        root=root,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        y_train=np.asarray(y_train),
        y_val=np.asarray(y_val),
        y_test=np.asarray(y_test),
    )


def make_gtsrb_loaders(
    splits: GTSRBSplits,
    image_size: int,
    batch_size: int,
    augment_train: bool,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tf = build_transforms(image_size, augment=augment_train)
    eval_tf = build_transforms(image_size, augment=False)

    train_ds = GTSRBDataset(splits.root, splits.train_paths, splits.y_train, train_tf)
    val_ds = GTSRBDataset(splits.root, splits.val_paths, splits.y_val, eval_tf)
    test_ds = GTSRBDataset(splits.root, splits.test_paths, splits.y_test, eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
