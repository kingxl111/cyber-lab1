from __future__ import annotations

import os
import json
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from src.data import make_splits
from src.models import BaselineMLP
from src.utils import seed_everything, make_loader, train_model, evaluate, save_metrics

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
EPOCHS = 20


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    split = make_splits(
        csv_path=DATA_PATH,
        use_engineered_features=False,
        test_size=0.2,
        val_size=0.2,
        random_state=SEED,
        min_frequency=20,
    )

    train_loader = make_loader(split.X_train, split.y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_loader(split.X_val, split.y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = make_loader(split.X_test, split.y_test, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = split.X_train.shape[1]
    model = BaselineMLP(input_dim=input_dim).to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=split.y_train,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=None,
        epochs=EPOCHS,
        patience=5,
    )

    test_metrics, y_true, y_pred, y_prob = evaluate(model, test_loader, device)

    PLOTS_DIR = os.path.join(OUT_DIR, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_loss(history, os.path.join(PLOTS_DIR, "baseline_loss.png"), "Baseline: loss curves")
    plot_confusion_matrix(y_true, y_pred, os.path.join(PLOTS_DIR, "baseline_cm.png"), "Baseline: confusion matrix")
    plot_roc_curve(y_true, y_prob, os.path.join(PLOTS_DIR, "baseline_roc.png"), "Baseline: ROC curve")
    plot_pr_curve(y_true, y_prob, os.path.join(PLOTS_DIR, "baseline_pr.png"), "Baseline: Precision-Recall curve")


    print("\nTEST METRICS (baseline)")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "baseline_mlp.pt"))
    history.to_csv(os.path.join(RESULTS_DIR, "baseline_history.csv"), index=False)
    save_metrics(test_metrics, os.path.join(RESULTS_DIR, "baseline_test_metrics.csv"))

    with open(os.path.join(RESULTS_DIR, "baseline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dim": input_dim,
                "epochs_trained": int(history.shape[0]),
                "test_metrics": test_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

if __name__ == "__main__":
    main()
