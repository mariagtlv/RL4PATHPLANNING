import torch
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'maxpool3x3',
    'conv3x3-bn-relu',
    'conv1x1-bn-relu'
]

DARTS = {'op': torch.Tensor([[0., 1., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.], [1., 0., 0.]]),
         'edge': [torch.Tensor([[0., 1.]]), torch.Tensor([[0., 0., 1.]]), torch.Tensor([[0., 0., 1., 0.]]), torch.Tensor([[0., 0., 0., 1., 0.]])]}

PDARTS = {'op': torch.Tensor([[0., 1., 0.], [1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.]]),
          'edge': [torch.Tensor([[0., 1.]]), torch.Tensor([[1., 0., 0.]]), torch.Tensor([[0., 0., 0., 1.]]), torch.Tensor([[0., 0., 0., 1., 0.]])]}

SDARTS = {'op': torch.Tensor([[0., 1., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 1.], [0., 1., 0.]]),
          'edge': [torch.Tensor([[0., 1.]]), torch.Tensor([[1., 0., 0.]]), torch.Tensor([[1., 0., 0., 0.]]), torch.Tensor([[0., 1., 0., 0., 0.]])]}

ADVRUSH = {'op': torch.Tensor([[0., 1., 0.], [1., 0., 0.], [1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]),
           'edge': [torch.Tensor([[0., 1.]]), torch.Tensor([[1., 0., 0.]]), torch.Tensor([[0., 0., 0., 1.]]), torch.Tensor([[0., 0., 0., 1., 0.]])]}

REP_DARTS = {'op': torch.Tensor([[1., 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [1., 0., 0.]]),
             'edge': [torch.Tensor([[0., 1.]]), torch.Tensor([[0., 0., 1.]]), torch.Tensor([[0., 1., 0., 0.]]), torch.Tensor([[0., 0., 0., 1., 0.]])]}