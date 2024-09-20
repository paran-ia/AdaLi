import torch
import torch.nn as nn
from myutils.common import heaviside
from config import AdaLiConfig
def piecewise_quadratic_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    x_abs = x.abs()
    mask = (x_abs > (1 / alpha))
    grad_x = (grad_output * (- (alpha ** 2) * x_abs + alpha)).masked_fill_(mask, 0)
    return grad_x, None


class piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha

        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_quadratic_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class PiecewiseQuadratic(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        AdaLiConfig['Left'][0] = 1.
        AdaLiConfig['Right'][0] = 1.


    def forward(self, x: torch.Tensor):
        return piecewise_quadratic.apply(x, self.alpha)

if __name__ == '__main__':
    # data = torch.rand(3,3).to("cuda:0").requires_grad_(True)
    # PiecewiseQuadratic()(data).sum().backward()
    # print(data.grad)
    d= torch.tensor(1.1)
    print(PiecewiseQuadratic()(d))