import torch
from torch.autograd import Variable
import torch.nn.functional as F


### PGD ###
distance_normal = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0]])

distance_reduction = torch.Tensor([[0, 0, 0, 1, 0, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0]])
distance_normal = Variable(distance_normal.cuda(), requires_grad=True)
distance_reduction = Variable(distance_reduction.cuda(), requires_grad=True)


### FGSM + PGD ###
# distance_normal = torch.Tensor([[0, 0, 0, 0, 1, 0, 0, 1],
#                   [0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 1],
#                   [0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 1],
#                   [0, 0, 0, 0, 0, 1, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 1],
#                   [0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0]])
#
# distance_reduction = torch.Tensor([[0, 1, 0, 1, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 1, 0, 0],
#                    [0, 1, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 1],
#                    [0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 1, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 1, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 1, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0]])
# distance_normal = Variable(distance_normal.cuda(), requires_grad=True)
# distance_reduction = Variable(distance_reduction.cuda(), requires_grad=True)


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def calculate_KL(alpha, distance):
    KL = []
    for i in range(len(alpha)):
        mean_alpha = torch.mean(alpha[i])
        std_alpha = torch.std(alpha[i])
        dist_alpha = torch.distributions.normal.Normal(mean_alpha, std_alpha)

        mean_distance = torch.mean(distance[i]) + 0.0001
        std_distance = torch.std(distance[i]) + 0.0001
        dist_distance = torch.distributions.normal.Normal(mean_distance, std_distance)

        KL.append(torch.distributions.kl_divergence(dist_alpha, dist_distance))

    for i in range(len(KL)):
        KL[i] = KL[i] / max(KL)
    KL_avg = sum(KL)/len(KL)

    return KL_avg


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def step(self, input_valid, target_valid):
    self.optimizer.zero_grad()
    self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    _, loss = self.model._loss(input_valid, target_valid)

    alphas_normal = F.softmax(self.model.alphas_normal, dim=-1)
    alphas_reduce = F.softmax(self.model.alphas_reduce, dim=-1)

    ### Euclidean distance ###
    distance_loss_normal = 0.01 * torch.sum((alphas_normal - distance_normal) ** 2)
    distance_loss_reduction = 0.01 * torch.sum((alphas_reduce - distance_reduction) ** 2)

    ### cosine distance ###
    # distance_loss_normal = 0.01 * torch.mean(torch.cosine_similarity(alphas_normal, distance_normal, dim=1))
    # distance_loss_reduction = 0.01 * torch.mean(torch.cosine_similarity(alphas_reduce, distance_reduction, dim=-1))

    ### KL divergence ###
    # distance_loss_normal = 0.01 * calculate_KL(alphas_normal, distance_normal)
    # distance_loss_reduction = 0.01 * calculate_KL(alphas_reduce, distance_reduction)

    loss += distance_loss_normal + distance_loss_reduction
    loss.backward()
