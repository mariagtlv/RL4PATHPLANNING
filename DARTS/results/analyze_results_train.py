import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "eval_metrics.csv"
OUT_DIR = Path("plots_eval")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, sep=";")

num_cols = [
    "train_acc", "train_loss",
    "valid_acc", "valid_loss",
    "f1_macro", "num_params",
    "total_time"
]

for col in num_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

df["epoch"] = df["epoch"].astype(int)

df = df.sort_values("epoch")

plt.figure()
plt.plot(df["epoch"], df["train_acc"], label="Training accuracy")
plt.plot(df["epoch"], df["valid_acc"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and validation accuracy over epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_vs_epoch.png")
plt.close()

plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="Training loss")
plt.plot(df["epoch"], df["valid_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss over epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "loss_vs_epoch.png")
plt.close()

df["generalization_gap"] = df["train_acc"] - df["valid_acc"]

plt.figure()
plt.plot(df["epoch"], df["generalization_gap"])
plt.xlabel("Epoch")
plt.ylabel("Train - Validation accuracy (%)")
plt.title("Generalization gap over epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "generalization_gap.png")
plt.close()

plt.figure()
plt.plot(df["epoch"], df["f1_macro"])
plt.xlabel("Epoch")
plt.ylabel("F1-macro")
plt.title("F1-macro score over epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "f1_macro_vs_epoch.png")
plt.close()

plt.figure()
plt.scatter(df["valid_loss"], df["valid_acc"], alpha=0.7)
plt.xlabel("Validation loss")
plt.ylabel("Validation accuracy (%)")
plt.title("Validation accuracy vs validation loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "valid_acc_vs_loss.png")
plt.close()
