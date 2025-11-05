import argparse
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchattacks
from model import TinyNetwork
from genotypes import Structure
import arch_sampled


parser = argparse.ArgumentParser(description='PyTorch CIFAR Clean Evaluation')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--test-batch-size', type=int, default=400, metavar='N', help='input batch size for testing (default: 200)')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
args = parser.parse_args()


torch.cuda.set_device(args.gpu)
cudnn.benchmark = True


transform_list = [transforms.ToTensor()]
transform_test = transforms.Compose(transform_list)
testset = torchvision.datasets.CIFAR10(root='/home/yuqi/data', train=False, download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)


def cal_acc(model, X, y):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    return err


def eval_std_acc(model, test_loader):
    model.eval()
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural = cal_acc(model, X, y)
        natural_err_total += err_natural
        print('batch err: ', natural_err_total)

    print('natural_err_total: ', natural_err_total)
    return natural_err_total


def eval_adv_acc(model, test_loader, type):
    model.eval()
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        if type == 'FGSM':
            attack = torchattacks.FGSM(model, eps=8/255)
        if type == 'PGD20':
            attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
        if type == 'CW':
            attack = torchattacks.CW(model, c=0.5, steps=100)
        if type == 'APGD':
            attack = torchattacks.APGD(model, eps=8/255, steps=20)
        adv_images = attack(data, target)
        X, y = Variable(adv_images, requires_grad=True), Variable(target)
        err_natural = cal_acc(model, X, y)
        natural_err_total += err_natural
        print('batch err: ', natural_err_total)

    print('natural_err_total: ', natural_err_total)
    return natural_err_total


def main():
    op = ['nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'none']
    code = [1, 1, 1, 3, 3, 0] #REP
    # code = [3, 3, 3, 3, 3, 3] #DARTS
    # code = [1, 3, 1, 3, 3, 3] #PDARTS
    # code = [0, 2, 1, 0, 2, 2] #SDARTS
    # code = [2, 3, 1, 3, 0, 2] #ADVRUSH
    genotype = Structure(
        [
            ((op[code[0]], 0),),
            ((op[code[1]], 0), (op[code[2]], 1)),
            ((op[code[3]], 0), (op[code[4]], 1), (op[code[5]], 2))
        ]
    )
    model = TinyNetwork(C=24, N=5, genotype=genotype, num_classes=10)
    model = model.cuda()
    model.load_state_dict(torch.load('./model.pt', map_location=torch.device('cuda:' + str(args.gpu))))
    eval_adv_acc(model, test_loader, 'FGSM')
    eval_adv_acc(model, test_loader, 'PGD20')
    eval_adv_acc(model, test_loader, 'APGD')
    eval_adv_acc(model, test_loader, 'CW')


if __name__ == '__main__':
    main()