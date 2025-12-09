import os
import os.path as osp
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import cat
import pickle
from sklearn.metrics import f1_score
import pandas as pd

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from utils import gen_uniform_60_20_20_split, save_load_split
from torch_geometric.utils import add_self_loops
from torch_geometric.datasets import Planetoid

parser = argparse.ArgumentParser("sane-train-search")

# ---------------- USER ARGS ----------------
parser.add_argument('--data', type=str, default='Cora')
parser.add_argument('--record_time', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=0.025)
parser.add_argument('--learning_rate_min', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--grad_clip', type=float, default=5)
parser.add_argument('--epsilon', type=float, default=0.0)
parser.add_argument('--unrolled', action='store_true', default=False)
parser.add_argument('--arch_learning_rate', type=float, default=3e-4)
parser.add_argument('--arch_weight_decay', type=float, default=1e-3)
parser.add_argument('--with_conv_linear', type=bool, default=False)
parser.add_argument('--fix_last', type=bool, default=False)

args = parser.parse_args()

def main():
    global device
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)

    print("args = ", args.__dict__)

    if args.data == 'Cora':
        dataset = Planetoid('/home/yuqi/data/', 'Cora')
    elif args.data == 'CiteSeer':
        dataset = Planetoid('/home/yuqi/data/', 'CiteSeer')

    raw_dir = dataset.raw_dir
    data = dataset[0]
    data = save_load_split(data, raw_dir, 1, gen_uniform_60_20_20_split)

    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))

    hidden_size = 32

    criterion = nn.CrossEntropyLoss().cuda()
    model = Network(
        criterion, dataset.num_features, dataset.num_classes,
        hidden_size, epsilon=args.epsilon, args=args
    ).cuda()

    print("param size = %fMB" % utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs),
        eta_min=args.learning_rate_min
    )

    architect = Architect(model, args)

    metrics = []
    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        lr = scheduler.get_lr()[0]
        genotype = model.genotype()

        train_acc, train_loss = train_trans(data, model, architect, criterion, optimizer, lr)
        valid_acc, valid_loss, valid_f1 = infer_trans(data, model, criterion)
        test_acc, test_loss, test_f1 = infer_trans(data, model, criterion, test=True)

        scheduler.step()
        explore_num = model.explore_num

        metrics.append({
            "timestamp": pd.Timestamp.now(),
            "epoch": epoch,
            "dataset": args.data,
            "seed": args.seed,
            "lr": lr,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "valid_acc": valid_acc,
            "valid_loss": valid_loss,
            "valid_f1": valid_f1,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "test_f1": test_f1,
            "params_MB": utils.count_parameters_in_MB(model),
            "genotype_normal": str(genotype.normal),
            "genotype_reduce": str(genotype.reduce),
            "genotype_full": str(genotype),
            "explore_num": explore_num,
            "epoch_time_sec": time.time() - epoch_start,
        })

        print(f"[Epoch {epoch}] train_acc={train_acc:.4f}, valid_acc={valid_acc:.4f}, test_acc={test_acc:.4f}")

    total_time = time.time() - total_start
    for m in metrics:
        m["total_time_sec"] = total_time

    df = pd.DataFrame(metrics)
    out_path = f"rep_gnn_results.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nMetrics saved to: {out_path}\n")

    return genotype


def train_trans(data, model, architect, criterion, optimizer, lr):
    model.train()

    target = data.y[data.train_mask].to(device)

    architect.step(data.to(device), lr, optimizer, unrolled=args.unrolled)

    logits = model(data.to(device))
    input_ = logits[data.train_mask]

    optimizer.zero_grad()
    loss = criterion(input_, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, _ = utils.accuracy(input_, target, topk=(1, 3))

    return prec1.item(), loss.item()


def infer_trans(data, model, criterion, test=False):
    model.eval()

    with torch.no_grad():
        logits = model(data.to(device))

    if test:
        mask = data.test_mask
    else:
        mask = data.val_mask

    preds = logits[mask].argmax(dim=1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    loss = criterion(logits[mask], data.y[mask].to(device)).item()
    acc, _ = utils.accuracy(logits[mask], data.y[mask].to(device), topk=(1, 3))
    f1 = f1_score(labels, preds, average="macro")

    return acc.item(), loss, f1


def run_by_seed():
    print(f"Searching architectures on {args.data}...")
    genotype = main()
    print(f"\nFinal genotype: {genotype}\n")


if __name__ == "__main__":
    run_by_seed()
