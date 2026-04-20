from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_loss(history, save_path: str, title: str):
    _ensure_dir(save_path)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_accuracy(history, save_path: str, title: str):
    if "train_acc" not in history.columns or "val_acc" not in history.columns:
        return

    _ensure_dir(save_path)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_acc"], label="Train acc")
    plt.plot(history["epoch"], history["val_acc"], label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str,
    num_classes: int,
    normalize: bool = True,
):
    _ensure_dir(save_path)

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    size = max(8, int(num_classes * 0.25))
    plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation="nearest", cmap="viridis", aspect="auto")
    plt.title(title)
    plt.colorbar()
    tick_step = max(1, num_classes // 20)
    ticks = np.arange(0, num_classes, tick_step)
    plt.xticks(ticks, [str(t) for t in ticks], rotation=90, fontsize=7)
    plt.yticks(ticks, [str(t) for t in ticks], fontsize=7)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str,
    num_classes: int,
):
    _ensure_dir(save_path)

    labels = list(range(num_classes))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    plt.figure(figsize=(max(8, num_classes * 0.3), 4))
    plt.bar(labels, per_class)
    plt.xlabel("Class id")
    plt.ylabel("F1-score")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
