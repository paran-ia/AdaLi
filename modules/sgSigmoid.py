import torch
import torch.nn as nn
from myutils.common import heaviside
from config import AdaLiConfig

def sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1. - sgax) * sgax * alpha, None

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class Sigmoid(nn.Module):
    def __init__(self, alpha = 1):
        super().__init__()
        self.alpha = alpha
        AdaLiConfig['Left'][0]=float('inf')
        AdaLiConfig['Right'][0] = float('inf')


    def forward(self, x: torch.Tensor):
        return sigmoid.apply(x, self.alpha)



if __name__ == '__main__':
    data = torch.rand(3,3).to("cuda:0").requires_grad_(True)
    Sigmoid()(data).sum().backward()
    print(data.grad)