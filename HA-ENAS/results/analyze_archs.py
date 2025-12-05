import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA

DEFAULT_RATIO = [3, 4, 6, 3]
GEN_MAP = {0: "Conv", 1: "Transformer", 2: "MLP"}
TOP_K = 10
OUTDIR = "analysis_output"
os.makedirs(OUTDIR, exist_ok=True)

def parse_decimal(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    s = s.replace(" ", "")
    if s.count(",") and s.count(".") and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def parse_genes(gstr):
    if pd.isna(gstr):
        return []
    s = str(gstr).strip()
    s = s.strip("[]")
    s = s.replace(".", " ")
    parts = re.split(r'[,\s]+', s)
    parts = [p for p in parts if p != ""]
    try:
        return [int(float(p)) for p in parts]
    except:
        return []

def decode_arch(genes, ratio=DEFAULT_RATIO):
    stages = []
    pos = 0
    for i, r in enumerate(ratio, start=2):
        stage_genes = genes[pos:pos + r]
        pos += r
        stage_types = [GEN_MAP.get(g, f"UNK({g})") for g in stage_genes]
        stages.append((f"c{i}", stage_types))
    s = "; ".join(f"{name}:{types}" for name, types in stages)
    return {"stages": stages, "readable": s}

def nondominated(points):
    pts = np.array(points)
    N = pts.shape[0]
    is_nd = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_nd[i]:
            continue
        for j in range(N):
            if i == j:
                continue
            if (pts[j,0] >= pts[i,0] and pts[j,1] >= pts[i,1]) and (pts[j,0] > pts[i,0] or pts[j,1] > pts[i,1]):
                is_nd[i] = False
                break
    return is_nd

def main(csv_path, ratio=DEFAULT_RATIO):
    print("Loading:", csv_path)
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            hdr = f.readline()
        sep = "\t" if "\t" in hdr else ";"
        df = pd.read_csv(csv_path, sep=sep, dtype=str)
    except Exception as e:
        print("Error reading CSV:", e)
        return
    
    cols = [c.lower() for c in df.columns]
    def find_col(possible):
        for p in possible:
            for c in df.columns:
                if c.lower() == p:
                    return c
        return None

    id_col = find_col(["id"])
    genes_col = find_col(["genes", "gene"])
    clean_col = find_col(["clean_acc", "clean"])
    black_col = find_col(["blackbox_acc", "blackbox", "black_acc"])

    if genes_col is None:
        print("No 'genes' column detected. Columns:", df.columns.tolist())
        return

    df["clean_acc_f"] = df[clean_col].apply(parse_decimal) if clean_col else np.nan
    df["black_acc_f"] = df[black_col].apply(parse_decimal) if black_col else np.nan

    df["genes_list"] = df[genes_col].apply(parse_genes)
    df["L"] = df["genes_list"].apply(len)
    print("Detected gene lengths (unique):", df["L"].unique())

    decoded = df["genes_list"].apply(lambda g: decode_arch(g, ratio=ratio))
    df["arch_readable"] = decoded.apply(lambda d: d["readable"])
    df["stages"] = decoded.apply(lambda d: ";".join([f"{n}={','.join(t)}" for n,t in d["stages"]]))

    df["balanced"] = df[["clean_acc_f", "black_acc_f"]].mean(axis=1)

    df_sorted_clean = df.sort_values(by="clean_acc_f", ascending=False).reset_index(drop=True)
    df_sorted_black = df.sort_values(by="black_acc_f", ascending=False).reset_index(drop=True)
    df_sorted_bal = df.sort_values(by="balanced", ascending=False).reset_index(drop=True)

    df_sorted_clean.head(TOP_K).to_csv(os.path.join(OUTDIR, "top_clean.csv"), index=False)
    df_sorted_black.head(TOP_K).to_csv(os.path.join(OUTDIR, "top_blackbox.csv"), index=False)
    df_sorted_bal.head(TOP_K).to_csv(os.path.join(OUTDIR, "top_balanced.csv"), index=False)
    df_sorted_bal.to_csv(os.path.join(OUTDIR, "best_models.csv"), index=False)

    print("Saved top_k csvs in", OUTDIR)

    pts = df[["clean_acc_f", "black_acc_f"]].fillna(-1).to_numpy()
    mask = nondominated(pts)
    df["pareto"] = mask
    df[df["pareto"]].to_csv(os.path.join(OUTDIR, "pareto.csv"), index=False)

    plt.figure(figsize=(7,6))
    plt.scatter(df["clean_acc_f"], df["black_acc_f"], alpha=0.6, label="architectures")
    pareto_df = df[df["pareto"]]
    plt.scatter(pareto_df["clean_acc_f"], pareto_df["black_acc_f"], color="red", label="Pareto front")
    plt.xlabel("clean_acc (%)")
    plt.ylabel("blackbox_acc (%)")
    plt.title("Clean vs Blackbox accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTDIR, "clean_vs_blackbox_scatter.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    df["clean_acc_f"].hist(bins=30)
    plt.xlabel("clean_acc (%)")
    plt.subplot(1,2,2)
    df["black_acc_f"].hist(bins=30)
    plt.xlabel("blackbox_acc (%)")
    plt.suptitle("Distributions")
    plt.savefig(os.path.join(OUTDIR, "distributions.png"), dpi=200)
    plt.close()

    maxL = df["genes_list"].map(len).max()
    df_good = df[df["L"] == sum(ratio)].copy()
    if df_good.shape[0] == 0:
        print("Warning: no rows with expected gene length. Skipping heatmap and PCA.")
    else:
        pos_counts = np.zeros((maxL, 3), dtype=int)
        for gl in df_good["genes_list"]:
            for i, g in enumerate(gl):
                if 0 <= g <= 2:
                    pos_counts[i, g] += 1
        pos_pct = pos_counts / pos_counts.sum(axis=1, keepdims=True)
        plt.figure(figsize=(10,4))
        plt.imshow(pos_pct.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label="fraction")
        plt.yticks([0,1,2], ["Conv","Transformer","MLP"])
        plt.xlabel("Gene position (0-index)")
        plt.title("Gene type fraction per position")
        plt.savefig(os.path.join(OUTDIR, "gene_position_heatmap.png"), dpi=200)
        plt.close()

        X = []
        for gl in df_good["genes_list"]:
            vec = []
            for g in gl:
                v = [0,0,0]
                if 0 <= g <= 2:
                    v[g] = 1
                vec.extend(v)
            X.append(vec)
        X = np.array(X)
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        plt.figure(figsize=(7,6))
        plt.scatter(X2[:,0], X2[:,1], alpha=0.6)
        plt.title("PCA 2D of architectures (one-hot genes)")
        plt.savefig(os.path.join(OUTDIR, "pca_archs.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to your CSV/TSV with columns id,timestamp,genes,clean_acc,blackbox_acc")
    parser.add_argument("--ratio", nargs="+", type=int, default=DEFAULT_RATIO,
                        help="Stage block counts, e.g. --ratio 3 4 6 3")
    args = parser.parse_args()
    main(args.csv, ratio=args.ratio)
