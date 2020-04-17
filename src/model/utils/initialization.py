import torch
import numpy as np
from torch import nn


def enable_reproducibility(seed=5697147651587154485):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def no_grad(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

def no_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def no_parameters(model):
    return sum(p.numel() for p in model.parameters())

def model_size(model):
    return sum(p.element_size() * p.nelement() for p in model.parameters())