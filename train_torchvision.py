from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torchvision import models

from src.plots import plot_loss, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from src.tabular_torchvision import make_torchvision_splits, make_torchvision_loaders
from src.utils import seed_everything, train_model, save_metrics


DATA_PATH = os.path.join("data", "customer_booking.csv")
OUT_DIR = os.path.join("outputs")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(OUT_DIR, "results")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def build_model(name: str):
    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    if name == "vit_b_16":
        model = models.vit_b_16(weights=None, image_size=64)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
        return model

    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    y_true, y_prob = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        y_true.extend(y.numpy().tolist())
        y_prob.extend(prob.tolist())

    return np.array(y_true), np.array(y_prob)


def metrics_at_threshold(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics, y_pred


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thr = 0.5
    best_f1 = -1.0
    best_metrics = None

    for thr in thresholds:
        metrics, _ = metrics_at_threshold(y_true, y_prob, thr)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thr = float(thr)
            best_metrics = metrics

    return best_thr, best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18", "vit_b_16"], required=True)
    parser.add_argument("--variant", choices=["baseline", "improved"], required=True)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    use_engineered_features = args.variant == "improved"
    min_frequency = 20 if args.variant == "baseline" else 10
    batch_size = 64 if args.model == "resnet18" else 32
    noise_std = 0.01 if args.variant == "baseline" else 0.02
    epochs = 8 if args.model == "resnet18" else 6
    patience = 3 if args.variant == "baseline" else 4

    split = make_torchvision_splits(
        csv_path=DATA_PATH,
        use_engineered_features=use_engineered_features,
        random_state=42,
        min_frequency=min_frequency,
    )

    train_loader, val_loader, test_loader = make_torchvision_loaders(
        split=split,
        batch_size=batch_size,
        noise_std=noise_std,
    )

    model = build_model(args.model).to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=split.y_train,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.variant == "baseline":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        epochs=epochs,
        patience=patience,
    )

    y_val_true, y_val_prob = predict_proba(model, val_loader, device)
    best_threshold, val_best_metrics = find_best_threshold(y_val_true, y_val_prob)

    y_test_true, y_test_prob = predict_proba(model, test_loader, device)
    test_metrics, y_test_pred = metrics_at_threshold(y_test_true, y_test_prob, best_threshold)

    run_name = f"{args.model}_{args.variant}"

    print(f"\nBest threshold on validation: {best_threshold:.2f}")
    print("Validation metrics at best threshold:")
    for k, v in val_best_metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nTEST METRICS ({run_name})")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{run_name}.pt"))
    history.to_csv(os.path.join(RESULTS_DIR, f"{run_name}_history.csv"), index=False)
    save_metrics(test_metrics, os.path.join(RESULTS_DIR, f"{run_name}_test_metrics.csv"))

    with open(os.path.join(RESULTS_DIR, f"{run_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "variant": args.variant,
                "best_threshold": best_threshold,
                "validation_metrics_at_best_threshold": val_best_metrics,
                "test_metrics": test_metrics,
                "epochs_trained": int(history.shape[0]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_loss(history, os.path.join(PLOTS_DIR, f"{run_name}_loss.png"), f"{run_name}: loss curves")
    plot_confusion_matrix(y_test_true, y_test_pred, os.path.join(PLOTS_DIR, f"{run_name}_cm.png"), f"{run_name}: confusion matrix")
    plot_roc_curve(y_test_true, y_test_prob, os.path.join(PLOTS_DIR, f"{run_name}_roc.png"), f"{run_name}: ROC curve")
    plot_pr_curve(y_test_true, y_test_prob, os.path.join(PLOTS_DIR, f"{run_name}_pr.png"), f"{run_name}: Precision-Recall curve")


if __name__ == "__main__":
    main()
