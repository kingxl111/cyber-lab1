from __future__ import annotations

import os
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from src.data import make_splits
from src.models import ImprovedMLP
from src.utils import seed_everything, make_loader, train_model, save_metrics

from src.plots import (
    plot_loss,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
)


DATA_PATH = os.path.join("data", "customer_booking.csv")
OUT_DIR = os.path.join("outputs")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(OUT_DIR, "results")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 256
EPOCHS = 30


def predict_proba(model, loader, device):
    model.eval()
    y_true = []
    y_prob = []

    with torch.no_grad():
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
    best_threshold = 0.5
    best_f1 = -1.0
    best_metrics = None

    for thr in thresholds:
        metrics, _ = metrics_at_threshold(y_true, y_prob, thr)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(thr)
            best_metrics = metrics

    return best_threshold, best_metrics


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    split = make_splits(
        csv_path=DATA_PATH,
        use_engineered_features=True,
        test_size=0.2,
        val_size=0.2,
        random_state=SEED,
        min_frequency=10,
    )

    train_loader = make_loader(split.X_train, split.y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_loader(split.X_val, split.y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = make_loader(split.X_test, split.y_test, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = split.X_train.shape[1]
    model = ImprovedMLP(input_dim=input_dim, dropout=0.35).to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=split.y_train,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
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
        epochs=EPOCHS,
        patience=7,
    )

    # Подбор лучшего порога по validation set
    y_val_true, y_val_prob = predict_proba(model, val_loader, device)
    best_threshold, val_best_metrics = find_best_threshold(y_val_true, y_val_prob)

    print(f"\nBest threshold on validation: {best_threshold:.2f}")
    print("Validation metrics at best threshold:")
    for k, v in val_best_metrics.items():
        print(f"{k}: {v:.4f}")

    # Финальная оценка на test set с выбранным порогом
    y_test_true, y_test_prob = predict_proba(model, test_loader, device)
    test_metrics, y_test_pred = metrics_at_threshold(y_test_true, y_test_prob, best_threshold)

    print("\nTEST METRICS (improved, tuned threshold)")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "improved_mlp.pt"))
    history.to_csv(os.path.join(RESULTS_DIR, "improved_history.csv"), index=False)
    save_metrics(test_metrics, os.path.join(RESULTS_DIR, "improved_test_metrics.csv"))

    with open(os.path.join(RESULTS_DIR, "improved_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dim": input_dim,
                "epochs_trained": int(history.shape[0]),
                "best_threshold": best_threshold,
                "validation_metrics_at_best_threshold": val_best_metrics,
                "test_metrics": test_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        
    PLOTS_DIR = os.path.join(OUT_DIR, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_loss(history, os.path.join(PLOTS_DIR, "improved_loss.png"), "Improved: loss curves")
    plot_confusion_matrix(y_test_true, y_test_pred, os.path.join(PLOTS_DIR, "improved_cm.png"), "Improved: confusion matrix")
    plot_roc_curve(y_test_true, y_test_prob, os.path.join(PLOTS_DIR, "improved_roc.png"), "Improved: ROC curve")
    plot_pr_curve(y_test_true, y_test_prob, os.path.join(PLOTS_DIR, "improved_pr.png"), "Improved: Precision-Recall curve")



if __name__ == "__main__":
    main()
