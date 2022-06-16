import torch
import torch.nn as nn
import numpy as np
from .softargmax import softargmax2d


class MeanRadialError(nn.Module):
    def __init__(self, shape=(640, 480), scale=1e-4, device='cpu'):
        super().__init__()
        self.h, self.w = shape
        self.scale  =scale
        self.device = device

        self.indices_c, self.indices_r = np.meshgrid(
            np.linspace(0, 1, self.w),
            np.linspace(0, 1, self.h),
            indexing='xy'
        )

        self.indices_r = torch.tensor(np.reshape(self.indices_r, (-1, self.h * self.w)), device=device)
        self.indices_c = torch.tensor(np.reshape(self.indices_c, (-1, self.h * self.w)), device=device)


    def softargmax2d(self, input):        
        *_, h, w = input.shape
        assert self.h == h and self.w == w

        beta = 100
        input = input.reshape(*_, self.h * self.w)
        input = torch.nn.functional.softmax(beta * input, dim=-1)

        result_r = torch.sum((self.h - 1) * input * self.indices_r, dim=-1)
        result_c = torch.sum((self.w - 1) * input * self.indices_c, dim=-1)
        result = torch.stack([result_r, result_c], dim=-1)

        return result

    def forward(self, pred, target):
        pred = pred.to(self.device)
        target = target.to(self.device)

        pred = self.softargmax2d(pred)
        target = self.softargmax2d(target)

        loss = torch.sum(torch.pow(pred - target, 2), axis=-1)
        loss = torch.mean(torch.sqrt(loss))
        
        return loss * self.scale


def mean_radial_error(pred, target):
    if type(pred) == torch.Tensor:
        device = pred.device
    else:
        device = 'cpu'

    pred = softargmax2d(pred, device=device)
    target = softargmax2d(target, device=device)

    loss = torch.sum(torch.pow(pred - target, 2), axis=-1)
    loss = torch.mean(torch.sqrt(loss))

    return loss


if __name__ == '__main__':
    device='cpu'

    loss = MeanRadialError(shape=(2, 25, 25), scale=1, device=device)

    x = torch.zeros(size=(2,25,25), device=device)
    y = torch.zeros(size=(2,25,25), device=device)

    x[0][3][3] = 1
    x[1][5][5] = 1

    y[0][3][4] = 1
    y[1][5][6] = 1

    print(softargmax2d(x))
    print(x.shape)
    print(y.shape)
    print(loss.forward(x, y))