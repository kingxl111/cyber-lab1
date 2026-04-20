from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn

from src.data import NUM_CLASSES, compute_class_weights, make_loaders, make_splits
from src.models import SimpleCNN
from src.plots import (
    plot_accuracy,
    plot_confusion_matrix_multiclass,
    plot_loss,
    plot_per_class_f1,
)
from src.utils import evaluate_multiclass, save_metrics, seed_everything, train_model


DATA_ROOT = "."
OUT_DIR = "outputs"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(OUT_DIR, "results")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

for _d in (CKPT_DIR, RESULTS_DIR, PLOTS_DIR):
    os.makedirs(_d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["baseline", "improved"],
        required=True,
        help="baseline - чистая собственная CNN без техник; "
        "improved - с техниками из улучшенного бейзлайна (п.4f ТЗ).",
    )
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs if args.epochs is not None else (
        15 if args.variant == "baseline" else 25
    )
    dropout = 0.2 if args.variant == "baseline" else 0.45

    run_name = f"customcnn_{args.variant}"
    print(f"Device: {device}")
    print(f"Run: {run_name} | image_size={args.image_size} | "
          f"batch_size={args.batch_size} | epochs={epochs} | dropout={dropout}")

    splits = make_splits(args.data_root, val_size=0.15, seed=args.seed)
    train_loader, val_loader, test_loader = make_loaders(
        data_root=args.data_root,
        splits=splits,
        image_size=args.image_size,
        variant=args.variant,
        pretrained_stats=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = SimpleCNN(num_classes=NUM_CLASSES, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleCNN parameters: {n_params:,}")

    class_weights = compute_class_weights(splits.train_df).to(device)

    if args.variant == "baseline":
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = None
        patience = 4
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        patience = 6

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

    test_metrics, y_true, y_pred, _ = evaluate_multiclass(
        model, test_loader, device, NUM_CLASSES
    )

    print(f"\nTEST METRICS ({run_name})")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{run_name}.pt"))
    history.to_csv(os.path.join(RESULTS_DIR, f"{run_name}_history.csv"), index=False)
    save_metrics(test_metrics, os.path.join(RESULTS_DIR, f"{run_name}_test_metrics.csv"))

    with open(os.path.join(RESULTS_DIR, f"{run_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "SimpleCNN",
                "variant": args.variant,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "epochs_config": epochs,
                "epochs_trained": int(history.shape[0]),
                "num_parameters": n_params,
                "test_metrics": test_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_loss(history, os.path.join(PLOTS_DIR, f"{run_name}_loss.png"), f"{run_name}: loss")
    plot_accuracy(history, os.path.join(PLOTS_DIR, f"{run_name}_acc.png"), f"{run_name}: accuracy")
    plot_confusion_matrix_multiclass(
        y_true,
        y_pred,
        os.path.join(PLOTS_DIR, f"{run_name}_cm.png"),
        f"{run_name}: normalized confusion matrix",
        NUM_CLASSES,
    )
    plot_per_class_f1(
        y_true,
        y_pred,
        os.path.join(PLOTS_DIR, f"{run_name}_f1.png"),
        f"{run_name}: per-class F1",
        NUM_CLASSES,
    )


if __name__ == "__main__":
    main()
