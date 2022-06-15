import torch


def L1_loss(pred, target):
    loss = torch.mean(torch.abs(pred - target))
    return loss


def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2))
    return loss