import argparse
import utils
import torch
import torch.nn as nn
import torchvision.datasets as dset
import numpy as np
import genotypes
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score
import pandas as pd
import os
import time
from datetime import datetime

from model import Network
from loss import trades_loss, madry_loss


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--data', type=str, default='/home/yuqi/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation')
parser.add_argument('--num_steps', type=int, default=7, help='perturb number of steps')
parser.add_argument('--step_size', type=float, default=0.01, help='perturb step size')
parser.add_argument('--init_channels', type=int, default=24, help='num of init channels')
parser.add_argument('--layers', type=int, default=9, help='total number of layers')
parser.add_argument('--beta', type=float, default=6.0, help='regularization in TRADES')
parser.add_argument('--adv_loss', type=str, default='pgd', help='experiment name')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--output_weights', type=bool, default=True, help='Whether to use weights on the output nodes')
args = parser.parse_args()


CIFAR_CLASSES = 10

def save_metrics(record):
    os.makedirs("rep_results", exist_ok=True)
    path = "rep_results/rep_cnn_nb101_training_results.parquet"

    df_new = pd.DataFrame([record])

    if os.path.exists(path):
        df_old = pd.read_parquet(path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_parquet(path, index=False)
    print(f"[PARQUET] Guardada fila en {path}")


def fgsm_attack(model, images, labels, eps):
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed = images + eps * images.grad.sign()
    return torch.clamp(perturbed, 0, 1)


def pgd_attack(model, images, labels, eps=0.031, alpha=0.007, steps=20):
    ori = images.clone().detach()
    images = images.clone().detach()
    for _ in range(steps):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv = images + alpha * images.grad.sign()
        eta = torch.clamp(adv - ori, min=-eps, max=eps)
        images = torch.clamp(ori + eta, 0, 1).detach()
    return images


def main():
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss().cuda()

    genotype = genotypes.REP_DARTS

    model = Network(
        args.init_channels, CIFAR_CLASSES, args.layers,
        criterion, args.output_weights, search_space='3',
        steps=5, geno=genotype
    ).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    best_acc = 0.0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        adjust_learning_rate(optimizer, epoch)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_loss = train(train_queue, model, criterion, optimizer)

        valid_acc, valid_loss, f1, fgsm_acc, pgd_acc = evaluate(valid_queue, model, criterion)

        params = utils.count_parameters_in_MB(model)

        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,

            "clean_acc": valid_acc,
            "clean_loss": valid_loss,

            "train_acc": train_acc,
            "train_loss": train_loss,

            "fgsm_acc": fgsm_acc,
            "pgd_acc": pgd_acc,
            "robustness": (fgsm_acc + pgd_acc) / 2,

            "f1_score": f1,
            "execution_time_epoch": time.time() - epoch_start,
            "params": params,

            "genotype_normal": str(genotype.normal) if hasattr(genotype, "normal") else None,
            "genotype_reduce": str(genotype.reduce) if hasattr(genotype, "reduce") else None,
            "genotype_full": str(genotype),
        }

        save_metrics(record)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), './model.pt')

        print(f"[EPOCH {epoch}] acc={valid_acc:.3f} | best={best_acc:.3f} | f1={f1:.3f} | pgd={pgd_acc:.3f}")


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    model.train()

    for inp, target in train_queue:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()

        if args.adv_loss == 'pgd':
            loss = madry_loss(
                model, inp, target, optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps
            )
        else:
            loss = trades_loss(
                model, inp, target, optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta
            )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        logits = model(inp)
        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), inp.size(0))
        top1.update(prec1.item(), inp.size(0))

    return top1.avg, objs.avg


def evaluate(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    model.eval()

    all_preds = []
    all_targets = []

    fgsm_correct = 0
    pgd_correct = 0
    total = 0

    with torch.no_grad():
        for inp, target in valid_queue:
            inp = inp.cuda()
            target = target.cuda()

            logits = model(inp)
            loss = criterion(logits, target)
            prec1, _ = utils.accuracy(logits, target, topk=(1, 5))

            objs.update(loss.item(), inp.size(0))
            top1.update(prec1.item(), inp.size(0))

            preds = torch.argmax(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            adv_fgsm = fgsm_attack(model, inp.clone(), target, eps=args.epsilon)
            out_fgsm = model(adv_fgsm)
            fgsm_correct += (torch.argmax(out_fgsm, 1) == target).sum().item()

            adv_pgd = pgd_attack(model, inp.clone(), target, eps=args.epsilon, alpha=args.step_size, steps=20)
            out_pgd = model(adv_pgd)
            pgd_correct += (torch.argmax(out_pgd, 1) == target).sum().item()

            total += target.size(0)

    f1 = f1_score(all_targets, all_preds, average="macro")

    fgsm_acc = fgsm_correct / total
    pgd_acc = pgd_correct / total

    return top1.avg, objs.avg, f1, fgsm_acc, pgd_acc


def adjust_learning_rate(optimizer, epoch):
    lr = args.learning_rate
    if epoch >= 99:
        lr *= 0.1
    if epoch >= 149:
        lr *= 0.01
    for pg in optimizer.param_groups:
        pg['lr'] = lr


if __name__ == '__main__':
    main()
