from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join("outputs", "results")
PLOTS_DIR = os.path.join("outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


FILES = {
    "MLP baseline": os.path.join(RESULTS_DIR, "baseline_test_metrics.csv"),
    "MLP improved": os.path.join(RESULTS_DIR, "improved_test_metrics.csv"),
    "ResNet18 baseline": os.path.join(RESULTS_DIR, "resnet18_baseline_test_metrics.csv"),
    "ResNet18 improved": os.path.join(RESULTS_DIR, "resnet18_improved_test_metrics.csv"),
    "ViT-B/16 baseline": os.path.join(RESULTS_DIR, "vit_b_16_baseline_test_metrics.csv"),
    "ViT-B/16 improved": os.path.join(RESULTS_DIR, "vit_b_16_improved_test_metrics.csv"),
}


def main():
    rows = []
    for name, path in FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            row = df.iloc[0].to_dict()
            row["model"] = name
            rows.append(row)

    if not rows:
        raise RuntimeError("No result files found in outputs/results")

    results = pd.DataFrame(rows)
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    results = results[cols].sort_values(by="f1", ascending=False)

    results.to_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index=False)
    print(results.to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.bar(results["model"], results["f1"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("F1-score")
    plt.title("Model comparison by F1-score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "comparison_f1.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(results["model"], results["roc_auc"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("ROC-AUC")
    plt.title("Model comparison by ROC-AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "comparison_roc_auc.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
