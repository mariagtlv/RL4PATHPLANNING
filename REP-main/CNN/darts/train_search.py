import numpy as np
import torch
import utils
import time
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchattacks

from torch.autograd import Variable
from model_search import Network
from architect import Architect

import pandas as pd
import os
from datetime import datetime

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='dataset', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs') #50 for sample, 10 for search
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()


CIFAR_CLASSES = 10

#NUEVO: Guardar en .parquet la arquitectura encontrada con metricas y metadata
def save_architecture_record(exp_path, epoch, genotype, valid_acc_fgsm, valid_acc_pgd, params_count):
    parquet_path = os.path.join(exp_path, "rep_darts_results.parquet")

    record = {
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "genotype_normal": str(genotype.normal),
        "genotype_reduce": str(genotype.reduce),
        "genotype_full": str(genotype),
        "fgsm_acc": valid_acc_fgsm,
        "pgd_acc": valid_acc_pgd,
        "robustness": 0.5 * (valid_acc_fgsm + valid_acc_pgd),
        "params": params_count,
    }

    df_new = pd.DataFrame([record])

    if os.path.exists(parquet_path):
        df_old = pd.read_parquet(parquet_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_parquet(parquet_path, index=False)
    print(f"[PARQUET] Achieved architecture saved → {parquet_path}")

def main():
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print('gpu device: ', args.gpu)
    print("args: ", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10_search(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    search_cost_training = 0
    search_cost_validation = 0
    archs = []
    robustness = []
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()
        print('epoch: ', epoch, ' lr: ', lr)

        time1 = time.time()
        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer)
        print('train_acc: ', train_acc)
        scheduler.step()
        time2 = time.time()
        search_cost_training += time2 - time1
        print('total cost for training: ', search_cost_training)

        genotype = model.genotype()
        print('genotype: ', genotype)

        if genotype not in archs:
            archs.append(genotype)
            print('perform validation!')

            valid_acc_fgsm, valid_obj_fgsm = infer(valid_queue, model, criterion, type='FGSM')
            valid_acc_pgd, valid_obj_pgd = infer(valid_queue, model, criterion, type='PGD')

            robust = 0.5 * (valid_acc_fgsm + valid_acc_pgd)
            robustness.append(robust)

            params = sum(p.numel() for p in model.parameters())

            exp_path = "rep_results"
            os.makedirs(exp_path, exist_ok=True)

            save_architecture_record(
                exp_path=exp_path,
                epoch=epoch,
                genotype=genotype,
                valid_acc_fgsm=valid_acc_fgsm,
                valid_acc_pgd=valid_acc_pgd,
                params_count=params
            )

            print('[INFO] Saved genotype + metrics in Parquet.')

        print(archs)
        print(robustness)

        dist = [0] * 16
        for i in range(len(archs) - 1):
            arch1 = archs[i]
            arch2 = archs[i + 1]
            same_pri = 0
            for op in range(len(arch1.normal)):
                if arch2.normal[2 * (op // 2)] == arch1.normal[op] or arch2.normal[2 * (op // 2) + 1] == arch1.normal[op]:
                    same_pri += 1
            for op in range(len(arch1.reduce)):
                if arch2.reduce[2 * (op // 2)] == arch1.reduce[op] or arch2.reduce[2 * (op // 2) + 1] == arch1.reduce[op]:
                    same_pri += 1
            dist[16 - same_pri] += 1
        print('Distribution of Similarity:', dist)

        time3 = time.time()
        search_cost_validation += time3 - time2
        print('total cost for validation: ', search_cost_validation)


def train(train_queue, valid_queue, model, architect, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda(non_blocking=True)
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda(non_blocking=True)
        target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

        architect.step(input_search, target_search)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, type):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    model.if_sharing = True

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input.cuda(), requires_grad=True)
        target = target.cuda()

        if type == 'FGSM':
            attack = torchattacks.FGSM(model, eps=2/255)
        elif type == 'PGD':
            attack = torchattacks.PGD(model, eps=2/255, steps=4)

        adv_images = attack(input, target)
        X, y = Variable(adv_images, requires_grad=True), Variable(target)

        logits = model(X)
        loss = criterion(logits, y)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    model.if_sharing = False

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()