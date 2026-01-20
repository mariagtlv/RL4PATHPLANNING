import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv(
    "results/training_metrics.csv",
    sep=";",
    decimal=",",
    parse_dates=["timestamp"],
    dayfirst=True
)

numeric_cols = ["epoch", "train_acc", "train_loss", "f1", "params"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

df = df.sort_values("epoch")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "figure.figsize": (7, 4),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10
})

plt.figure()
plt.plot(df["epoch"], df["train_acc"], marker="o", label="Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("results/accuracy_vs_epoch.png", dpi=300)
plt.close()

plt.figure()
plt.plot(df["epoch"], df["train_loss"], marker="o", color="tab:red", label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_vs_epoch.png", dpi=300)
plt.close()

plt.figure()
plt.plot(df["epoch"], df["f1"], marker="o", color="tab:green", label="F1-score")
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.title("F1-score vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("results/f1_vs_epoch.png", dpi=300)
plt.close()

plt.figure()
plt.plot(df["timestamp"], df["train_acc"], marker="o", label="Train Accuracy")
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Time")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

plt.legend()
plt.tight_layout()
plt.savefig("results/accuracy_vs_time.png", dpi=300)
plt.close()

