from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data import make_splits


IMG_SIDE = 64
IMG_LEN = IMG_SIDE * IMG_SIDE


def vector_to_image(vec: np.ndarray, target_len: int = IMG_LEN) -> torch.Tensor:
    v = vec.astype(np.float32)

    if len(v) < target_len:
        v = np.pad(v, (0, target_len - len(v)))
    elif len(v) > target_len:
        v = v[:target_len]

    img = torch.tensor(v, dtype=torch.float32).view(1, IMG_SIDE, IMG_SIDE)
    img = img.repeat(3, 1, 1)
    return img


class TabularImageDataset(Dataset):
    def __init__(self, X, y, training: bool = False, noise_std: float = 0.0):
        self.X = X
        self.y = y.astype(np.int64)
        self.training = training
        self.noise_std = noise_std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = vector_to_image(self.X[idx])
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x = torch.clamp(x, -5.0, 5.0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def make_torchvision_splits(
    csv_path: str,
    use_engineered_features: bool,
    random_state: int = 42,
    min_frequency: int | None = None,
):
    return make_splits(
        csv_path=csv_path,
        use_engineered_features=use_engineered_features,
        test_size=0.2,
        val_size=0.2,
        random_state=random_state,
        min_frequency=min_frequency,
    )


def make_torchvision_loaders(
    split,
    batch_size: int,
    noise_std: float = 0.0,
):
    train_ds = TabularImageDataset(split.X_train, split.y_train, training=True, noise_std=noise_std)
    val_ds = TabularImageDataset(split.X_val, split.y_val, training=False)
    test_ds = TabularImageDataset(split.X_test, split.y_test, training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
