from torch import nn
from .mstcn import MSTCNLoss
from .ce import CrossEntropyLoss


def get_loss_function(criterion: str):
    if criterion == 'mstcn':
        return MSTCNLoss()
    if criterion == 'ce':
        return CrossEntropyLoss()
    raise ValueError(f'Unknown criterion: {criterion}.')
