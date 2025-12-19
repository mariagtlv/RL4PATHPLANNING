import re
import csv
from datetime import datetime

LOG_FILE = "DARTS/eval-EXP-20251218-123106/log.txt"
OUTPUT_CSV = "DARTS/results/training_metrics.csv"
RUN_TYPE = "train" 
GENOTYPE_NAME = "DARTS"  

epoch_re = re.compile(r"^(.*?) epoch (\d+) lr")
genotype_re = re.compile(r"genotype = (Genotype\(.*\))")
train_acc_re = re.compile(r"train_acc ([0-9.]+)")
valid_acc_re = re.compile(r"valid_acc ([0-9.]+)")
param_re = re.compile(r"param size = ([0-9.]+)MB")

def parse_time(ts):
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")

def parse_genotype(genotype_str):
    normal = re.search(r"normal=\[(.*?)\], normal_concat", genotype_str)
    reduce = re.search(r"reduce=\[(.*?)\], reduce_concat", genotype_str)
    return (
        normal.group(1) if normal else "",
        reduce.group(1) if reduce else ""
    )

rows = []
current = {}
params_mb = None
epoch_start_time = None

with open(LOG_FILE, "r") as f:
    for line in f:
        line = line.strip()

        m = param_re.search(line)
        if m:
            params_mb = float(m.group(1))

        m = epoch_re.search(line)
        if m:
            timestamp_str, epoch = m.groups()
            epoch_start_time = parse_time(timestamp_str)

            current = {
                "timestamp": timestamp_str,
                "epoch": int(epoch),
                "genotype_normal": None,
                "genotype_reduce": None,
                "genotype_full": None,
                "clean_acc": None,
                "clean_loss": None,
                "train_acc": None,
                "train_loss": None,
                "f1": None,
                "params": params_mb,
                "total_time_sec": None,
            }

            if RUN_TYPE == "train":
                current["genotype_full"] = GENOTYPE_NAME
                current["genotype_normal"] = GENOTYPE_NAME
                current["genotype_reduce"] = GENOTYPE_NAME

            continue

        if RUN_TYPE == "search":
            m = genotype_re.search(line)
            if m and current:
                g_full = m.group(1)
                g_normal, g_reduce = parse_genotype(g_full)
                current["genotype_full"] = g_full
                current["genotype_normal"] = g_normal
                current["genotype_reduce"] = g_reduce
                continue

        m = train_acc_re.search(line)
        if m and current:
            current["train_acc"] = float(m.group(1))
            continue

        m = valid_acc_re.search(line)
        if m and current:
            valid_acc = float(m.group(1))
            current["clean_acc"] = valid_acc

            if epoch_start_time:
                epoch_end_time = parse_time(line.split(" valid_acc")[0])
                current["total_time_sec"] = (
                    epoch_end_time - epoch_start_time
                ).total_seconds()

            rows.append(current)
            current = {}

if RUN_TYPE == "search":
    fieldnames = [
        "timestamp",
        "epoch",
        "genotype_normal",
        "genotype_reduce",
        "genotype_full",
        "train_acc",
        "clean_acc",
    ]
else:
    fieldnames = [
        "timestamp",
        "epoch",
        "genotype_normal",
        "genotype_reduce",
        "genotype_full",
        "clean_acc",
        "clean_loss",
        "train_acc",
        "train_loss",
        "f1",
        "params",
        "total_time_sec",
    ]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} epochs to {OUTPUT_CSV}")
