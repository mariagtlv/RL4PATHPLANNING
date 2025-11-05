import torch
from torch.autograd import Variable
import torch.nn.functional as F


V = torch.Tensor([[0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0]])
V = Variable(V.cuda(), requires_grad=True)


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(model.get_alphas(), lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                        weight_decay=args.arch_weight_decay)

  def step(self, input_valid, target_valid):
    self.optimizer.zero_grad()
    self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)

    alpha = F.softmax(self.model.arch_parameters, dim=-1)
    distance_loss = 0.01 * torch.sum((alpha - V) ** 2)

    loss += distance_loss
    loss.backward()