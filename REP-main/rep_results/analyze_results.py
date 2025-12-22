import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "rep_results/comparativa.csv"
OUTPUT_DIR = Path("rep_results/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

ARCHITECTURES = ["cnn_darts", "cnn_nb101", "cnn_nats", "gnn"]

df = pd.read_csv(
    CSV_PATH,
    sep=";",
    decimal=","
)

num_cols = ["fgsm_acc", "pgd_acc", "robustness", "params", "epoch"]
df[num_cols] = df[num_cols].apply(pd.to_numeric)

def pareto_front(df, x_col, y_col):
    points = df[[x_col, y_col]].values
    is_pareto = np.ones(len(points), dtype=bool)

    for i, (x_i, y_i) in enumerate(points):
        for j, (x_j, y_j) in enumerate(points):
            if j != i and x_j >= x_i and y_j >= y_i:
                if x_j > x_i or y_j > y_i:
                    is_pareto[i] = False
                    break

    return df[is_pareto]

for arch in ARCHITECTURES:
    df_arch = df[df["search_space"] == arch]

    if df_arch.empty:
        continue

    for metric in ["fgsm_acc", "pgd_acc", "robustness"]:
        plt.figure()
        plt.hist(df_arch[metric], bins=10)
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency")
        plt.title(f"{arch} - {metric} histogram")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{arch}_hist_{metric}.png")
        plt.close()

    pareto_df = pareto_front(df_arch, "fgsm_acc", "pgd_acc")

    plt.figure()
    plt.scatter(
        df_arch["fgsm_acc"],
        df_arch["pgd_acc"],
        label="Models",
        alpha=0.7
    )
    plt.scatter(
        pareto_df["fgsm_acc"],
        pareto_df["pgd_acc"],
        label="Pareto Front",
        s=80
    )

    plt.xlabel("FGSM Accuracy (%)")
    plt.ylabel("PGD Accuracy (%)")
    plt.title(f"{arch} - FGSM vs PGD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{arch}_fgsm_vs_pgd_pareto.png")
    plt.close()

print("Plots generated in:", OUTPUT_DIR.resolve())
