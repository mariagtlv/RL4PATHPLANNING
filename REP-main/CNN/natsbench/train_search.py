import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.utils
import utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import pandas as pd
import os
from datetime import datetime
import torchattacks
from sklearn.metrics import f1_score

from torch.autograd import Variable
from model_search import TinyNetworkDarts as Network
from architect import Architect

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--data', type=str, default='dataset', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

def save_metrics_record(record, exp_path):
    parquet_path = os.path.join(exp_path, "rep_cnn_natsbench_results.parquet")

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
    print('gpu device:', args.gpu)
    print("args:", args)

    exp_path = "rep_results"
    os.makedirs(exp_path, exist_ok=True)

    criterion = nn.CrossEntropyLoss().cuda()

    search_space = ['nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'none']

    model = Network(
        C=16,
        N=5,
        max_nodes=4,
        num_classes=10,
        search_space=search_space,
        criterion=criterion
    ).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        print(model.show_alphas())
        genotype = model.genotype()
        print("Genotype:", genotype)

        train_acc, train_loss = train(train_queue, valid_queue, model, criterion, optimizer, architect)

        valid_acc_clean, valid_loss_clean, clean_true, clean_pred = infer(valid_queue, model, criterion)

        fgsm_acc, fgsm_loss, _, _ = infer(valid_queue, model, criterion, attack_type="FGSM")

        pgd_acc, pgd_loss, _, _ = infer(valid_queue, model, criterion, attack_type="PGD")

        robustness = 0.5 * (fgsm_acc + pgd_acc)

        params = sum(p.numel() for p in model.parameters())

        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "genotype_normal": str(getattr(genotype, "normal", "-")),
            "genotype_reduce": str(getattr(genotype, "reduce", "-")),
            "genotype_full": str(genotype),
            "fgsm_acc":fgsm_acc,
            "pgd_acc":pgd_acc,
            "robustness":robustness,
            "params": params
        }

        save_metrics_record(record, exp_path)

        scheduler.step()

def train(train_queue, valid_queue, model, criterion, optimizer, architect):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        input_search, target_search = next(iter(valid_queue))
        architect.step(
            input_search.cuda(non_blocking=True),
            target_search.cuda(non_blocking=True)
        )

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


def infer(valid_queue, model, criterion, attack_type=None):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    all_true = []
    all_pred = []

    model.eval()

    for step, (input, target) in enumerate(valid_queue):

        input = Variable(input.cuda(), requires_grad=True)
        target = target.cuda()

        if attack_type == "FGSM":
            attack = torchattacks.FGSM(model, eps=2/255)
            adv = attack(input, target)
            X = adv
        elif attack_type == "PGD":
            attack = torchattacks.PGD(model, eps=2/255, steps=4)
            adv = attack(input, target)
            X = adv
        else:
            X = input  

        logits = model(X)
        loss = criterion(logits, target)

        pred = logits.argmax(dim=1)

        all_true.extend(target.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    return top1.avg, objs.avg, all_true, all_pred

if __name__ == '__main__':
    main()
