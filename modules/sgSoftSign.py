import torch
import torch.nn as nn
from myutils.common import heaviside
from config import AdaLiConfig

def soft_sign_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output / (2 * alpha * (1 / alpha + x.abs()).pow_(2)), None

class soft_sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return soft_sign_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class SoftSign(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
        AdaLiConfig['Left'][0]=float('inf')
        AdaLiConfig['Right'][0] = float('inf')


    def forward(self, x: torch.Tensor):
        return soft_sign.apply(x, self.alpha)


if __name__ == '__main__':
    data = torch.rand(3,3).to("cuda:0").requires_grad_(True)
    SoftSign()(data).sum().backward()
    print(data.grad)