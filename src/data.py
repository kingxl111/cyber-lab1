from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


NUM_CLASSES = 43

GTSRB_CLASS_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5 t", 11: "Right-of-way at next intersection",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Vehicles over 3.5 t prohibited", 17: "No entry", 18: "General caution",
    19: "Dangerous curve left", 20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed/passing limits", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End of no passing", 42: "End of no passing by vehicles over 3.5 t",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD = (0.2724, 0.2608, 0.2669)


class GTSRBDataset(Dataset):
    """
    GTSRB dataset wrapper. Rows come from Train.csv / Test.csv from the Kaggle version.

    CSV columns: Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path.
    Path is relative to `data_root` (e.g. "Train/20/00020_00000_00000.png").
    """

    def __init__(
        self,
        data_root: str,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        use_roi: bool = True,
    ):
        self.data_root = data_root
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_roi = use_roi

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["Path"])
        img = Image.open(img_path).convert("RGB")

        if self.use_roi:
            img = img.crop(
                (
                    int(row["Roi.X1"]),
                    int(row["Roi.Y1"]),
                    int(row["Roi.X2"]),
                    int(row["Roi.Y2"]),
                )
            )

        if self.transform is not None:
            img = self.transform(img)

        label = int(row["ClassId"])
        return img, label


def build_transforms(
    image_size: int,
    variant: str,
    pretrained_stats: bool,
) -> Tuple[Callable, Callable]:
    """
    variant:
        "baseline"  - минимальный препроцессинг
        "improved"  - с CV-аугментациями
    pretrained_stats:
        True  - ImageNet mean/std (для torchvision pretrained моделей)
        False - GTSRB mean/std (для собственной модели с нуля)

    Важно: НЕ используем HorizontalFlip - у многих знаков (повороты, "keep right"
    и т.д.) горизонтальное отражение меняет класс.
    """
    mean = IMAGENET_MEAN if pretrained_stats else GTSRB_MEAN
    std = IMAGENET_STD if pretrained_stats else GTSRB_STD

    eval_t = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if variant == "baseline":
        train_t = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return train_t, eval_t

    pad = max(2, int(image_size * 0.1))
    train_t = transforms.Compose(
        [
            transforms.Resize((image_size + pad, image_size + pad)),
            transforms.RandomCrop(image_size),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                shear=8,
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )
    return train_t, eval_t


@dataclass
class GTSRBSplits:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def make_splits(
    data_root: str,
    val_size: float = 0.15,
    seed: int = 42,
) -> GTSRBSplits:
    train_csv = pd.read_csv(os.path.join(data_root, "Train.csv"))
    test_csv = pd.read_csv(os.path.join(data_root, "Test.csv"))

    train_df, val_df = train_test_split(
        train_csv,
        test_size=val_size,
        random_state=seed,
        stratify=train_csv["ClassId"],
    )
    return GTSRBSplits(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_csv.reset_index(drop=True),
    )


def make_loaders(
    data_root: str,
    splits: GTSRBSplits,
    image_size: int,
    variant: str,
    pretrained_stats: bool,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_t, eval_t = build_transforms(image_size, variant, pretrained_stats)

    train_ds = GTSRBDataset(data_root, splits.train_df, transform=train_t)
    val_ds = GTSRBDataset(data_root, splits.val_df, transform=eval_t)
    test_ds = GTSRBDataset(data_root, splits.test_df, transform=eval_t)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def compute_class_weights(
    train_df: pd.DataFrame,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    counts = np.bincount(train_df["ClassId"].values, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
