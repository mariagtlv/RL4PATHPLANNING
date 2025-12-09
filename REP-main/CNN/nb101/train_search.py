import numpy as np
import torch
import utils
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import pandas as pd
import os
import time
from datetime import datetime
from sklearn.metrics import f1_score

from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=9, help='total number of layers')
parser.add_argument('--data', type=str, default='dataset', help='location of the data corpus')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--output_weights', type=bool, default=True, help='Whether to use weights on the output nodes')
args = parser.parse_args()


CIFAR_CLASSES = 10

def save_metrics_record(record, exp_path):
    parquet_path = os.path.join(exp_path, "rep_cnn_nb101_results.parquet")

    df_new = pd.DataFrame([record])

    if os.path.exists(parquet_path):
        df_old = pd.read_parquet(parquet_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_parquet(parquet_path, index=False)
    print(f"[PARQUET] Métrica guardada → {parquet_path}")

def main():
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print('gpu device: ', args.gpu)
    print("args: ", args)

    exp_path = "rep_results"
    os.makedirs(exp_path, exist_ok=True)

    criterion = nn.CrossEntropyLoss().cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers,
                    criterion, args.output_weights, search_space='3', steps=5).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch}, lr {lr}")
        print("genotype:", model.genotype())

        train_acc, train_loss = train(train_queue, valid_queue, model, architect, criterion, optimizer)
        print("train_acc:", train_acc)

        valid_acc, valid_loss, f1 = infer(valid_queue, model, criterion)
        print("valid_acc:", valid_acc, "F1:", f1)

        scheduler.step()

        params = sum(p.numel() for p in model.parameters())
        exec_time = time.time() - epoch_start

        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "learning_rate": lr,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "valid_acc": valid_acc,
            "valid_loss": valid_loss,
            "f1_score": f1,
            "execution_time_epoch": exec_time,
            "genotype": str(model.genotype()),
            "params": params
        }

        save_metrics_record(record, exp_path)


def train(train_queue, valid_queue, model, architect, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)

        architect.step(input_search, target_search)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average='macro')

    return top1.avg, objs.avg, f1


if __name__ == '__main__':
    main()
