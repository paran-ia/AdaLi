import torch
import torch.nn as nn
from config import AdaLiConfig
from myutils.common import heaviside


def adali_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float, beta: float):
    #边界已由handleConfig函数处理好
    #根据边界和alpha, beta更新梯度
    left = AdaLiConfig['Vth'] - AdaLiConfig['Left'][0]
    right = AdaLiConfig['Vth'] + AdaLiConfig['Right'][0]
    grad = torch.ones_like(x)
    mask = (x<left)|(x>right)
    grad = torch.where(x<AdaLiConfig['Vth'],grad*alpha / AdaLiConfig['Left'][0],grad*beta / AdaLiConfig['Right'][0]).masked_fill_(mask, 0)
    return grad_output * grad, None, None

class adali(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, beta):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
            ctx.beta = beta
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return adali_backward(grad_output, ctx.saved_tensors[0], ctx.alpha, ctx.beta)

class AdaLi(nn.Module):
    def __init__(self, alpha = 0.5, beta =0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor):
        return adali.apply(x, self.alpha, self.beta)



if __name__ == '__main__':
    data = torch.rand(5,5).to("cuda:0").requires_grad_(True)

    AdaLi()(data).sum().backward()
    print(data.grad)