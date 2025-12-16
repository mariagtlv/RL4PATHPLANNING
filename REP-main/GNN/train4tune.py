import os
import os.path as osp
import sys
import time
import glob
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch import cat
from sklearn.metrics import f1_score
import pandas as pd

from torch.autograd import Variable
from model import NetworkGNN as Network
from utils import gen_uniform_60_20_20_split, save_load_split
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops

parser = argparse.ArgumentParser("sane-train")
parser.add_argument('--data', type=str, default='Cora')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=0.25)
parser.add_argument('--learning_rate_min', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--grad_clip', type=float, default=5)
parser.add_argument('--with_linear', action='store_true', default=False)
parser.add_argument('--fix_last', type=bool, default=False)
train_args = parser.parse_args()

def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.cuda.set_device(train_args.gpu)
    torch.manual_seed(train_args.seed)
    cudnn.enabled=True
    cudnn.benchmark=True
    torch.cuda.manual_seed(train_args.seed)

    data_root = osp.join(os.getcwd(), "data")

    if train_args.data == 'Cora':
        dataset = Planetoid(data_root, 'Cora')
    elif train_args.data == 'CiteSeer':
        dataset = Planetoid(data_root, 'CiteSeer')
    else:
        raise ValueError(
            f"Unknown dataset '{train_args.data}'. "
            "Valid options are: 'Cora' or 'CiteSeer'."
        )

    genotype = 'gat_cos||gin||gat_generalized_linear||skip||skip||skip||l_concat'
    hidden_size = train_args.hidden_size

    raw_dir = dataset.raw_dir
    data = dataset[0]
    data = save_load_split(data, raw_dir, 1, gen_uniform_60_20_20_split)
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
    data.edge_index = edge_index

    num_features = dataset.num_features
    num_classes = dataset.num_classes

    criterion = nn.CrossEntropyLoss().cuda()
    model = Network(genotype, criterion, num_features, num_classes, hidden_size,
                    num_layers=train_args.num_layers, is_mlp=False, args=train_args).cuda()

    print("genotype = {}, param size = {}MB, args = {}".format(genotype, utils.count_parameters_in_MB(model), train_args.__dict__))

    params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.SGD(
        model.parameters(),
        train_args.learning_rate,
        momentum=train_args.momentum,
        weight_decay=train_args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs))

    metrics = [] 
    total_start = time.time()
    best_val_acc = best_test_acc = 0

    for epoch in range(train_args.epochs):
        epoch_start = time.time()

        train_acc, train_loss = train_trans(data, model, criterion, optimizer)
        scheduler.step()
        valid_acc, valid_loss, valid_f1 = infer_trans(data, model, criterion)
        test_acc, test_loss, test_f1 = infer_trans(data, model, criterion, test=True)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_test_acc = test_acc
            utils.save(model, './model.pt')

        metrics.append({
            "timestamp": pd.Timestamp.now(),
            "epoch": epoch,

            "genotype_normal": str(getattr(genotype, "normal", "-")),
            "genotype_reduce": str(getattr(genotype, "reduce", "-")),   
            "genotype_full": str(genotype),

            "clean_acc": valid_acc,
            "clean_loss": valid_loss,
            "train_acc": train_acc,
            "train_loss": train_loss,
            
            "f1": test_f1,

            "params": params,
            "total_time_sec": time.time() - epoch_start
        })

        print(f"[Epoch {epoch}] train_acc={train_acc:.4f}, valid_acc={valid_acc:.4f}, test_acc={test_acc:.4f}")

    total_time = time.time() - total_start
    for m in metrics:
        m["total_time_sec"] = total_time

    df = pd.DataFrame(metrics)
    df.to_parquet(f"rep_gnn_training_results.parquet", index=False)
    print(f"\nMetrics saved to rep_gnn_training_results.parquet")
    print('best validation acc: {}, best test acc: {}'.format(best_val_acc, best_test_acc))
    print('total training time: {:.2f}s'.format(total_time))


def train_trans(data, model, criterion, optimizer):
    model.train()
    target = data.y[data.train_mask].to(device)

    optimizer.zero_grad()
    logits = model(data.to(device))
    input_ = logits[data.train_mask]
    loss = criterion(input_, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
    optimizer.step()

    prec1, _ = utils.accuracy(input_, target, topk=(1, 3))
    return prec1.item(), loss.item()


def infer_trans(data, model, criterion, test=False):
    model.eval()
    with torch.no_grad():
        logits = model(data.to(device))

    mask = data.test_mask if test else data.val_mask
    input_ = logits[mask].to(device)
    target = data.y[mask].to(device)

    loss = criterion(input_, target).item()
    prec1, _ = utils.accuracy(input_, target, topk=(1, 3))

    preds = input_.argmax(dim=1).cpu().numpy()
    labels = target.cpu().numpy()
    f1 = f1_score(labels, preds, average="macro")

    return prec1.item(), loss, f1


if __name__ == '__main__':
    main()
