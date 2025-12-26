import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "search_metrics.csv"
OUT_DIR = Path("plots_search")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, sep=";")

for col in ["train_acc", "valid_acc"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

df["epoch"] = df["epoch"].astype(int)

df = df.sort_values("epoch")

plt.figure()
plt.plot(df["epoch"], df["train_acc"], label="Train accuracy")
plt.plot(df["epoch"], df["valid_acc"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Validation Accuracy during search")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_vs_epoch.png")
plt.close()

df["gap"] = df["train_acc"] - df["valid_acc"]

plt.figure()
plt.plot(df["epoch"], df["gap"])
plt.xlabel("Epoch")
plt.ylabel("Train - Validation Accuracy (%)")
plt.title("Generalization gap during search")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "generalization_gap.png")
plt.close()

plt.figure()
plt.scatter(df["train_acc"], df["valid_acc"], alpha=0.7)
plt.xlabel("Train accuracy (%)")
plt.ylabel("Validation accuracy (%)")
plt.title("Train vs Validation Accuracy (found architectures)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "train_vs_valid_scatter.png")
plt.close()

df["best_valid_so_far"] = df["valid_acc"].cummax()

plt.figure()
plt.plot(df["epoch"], df["best_valid_so_far"])
plt.xlabel("Epoch")
plt.ylabel("Best validation accuracy (%)")
plt.title("Best accumulated validation accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "best_valid_acc_over_time.png")
plt.close()

plt.figure()
plt.hist(df["valid_acc"], bins=20)
plt.xlabel("Validation accuracy (%)")
plt.ylabel("Frecuencia")
plt.title("Validation accuracy distribution")
plt.tight_layout()
plt.savefig(OUT_DIR / "valid_acc_histogram.png")
plt.close()

