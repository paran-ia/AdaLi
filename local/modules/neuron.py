import torch.nn as nn
import torch
from modules.adaptive_func import *
def spike_activation(x,Vth,wl,wr,ada):
    out_s = torch.gt(x,Vth)
    if ada == 'log':
        wl,wr = log_func(wl,wr,'+')
    elif ada == 'linear':
        wl,wr = linear_func(wl,wr,'+')
    elif ada == 'sqrt':
        wl,wr = sqrt_func(wl,wr,'+')
    if wl <= 0.2 or wr <= 0.2:
        raise ValueError("width can not be too small")
    k1 = 0.5/wl
    k2 = 0.5/wr
    out_bp = torch.clamp(x,min=Vth-wl, max=Vth+wr)
    out_bp = torch.where(((out_bp>=Vth-wl) & (out_bp<Vth)),out_bp*k1,torch.where(((out_bp>=Vth) & (out_bp<=Vth+wr)),out_bp*k2,out_bp))
    return (out_s.float() - out_bp).detach() + out_bp
def mem_update(neuron_type,x_in, mem, V_th, decay,wl,wr,reset,ada):
    if neuron_type == 'LIF':
        mem = mem * decay + x_in
    elif neuron_type == 'IF':
        mem = mem + x_in
    else:
        raise NotImplementedError("invalid neuron type")
    u = mem.clone()
    spike = spike_activation(mem/V_th,V_th,wl,wr,ada)
    if reset:
        mem = mem * (1 - spike)
    else:
        mem = torch.where(spike.bool(),mem-V_th,mem)
    return u, mem, spike

class LIF(nn.Module):
    def __init__(self, Vth=1, step=4, wl=0.5, wr =0.5,adaptive:str='none', reset:bool=True):
        super().__init__()
        self.step = step
        self.wl = wl
        self.wr = wr
        self.V_th = Vth
        self.reset=reset
        self.type='LIF'
        self.ada=adaptive
        self.u=[]#before fire a spike
        self.out=[]
    def forward(self, x):

        # if x.dim() != 5:
        #     raise TypeError(f'The input dim of spiking neurons should be 5, but got the dim of {x.dim()}')
        #
        if x.shape[0] != self.step:
            raise ValueError(f'SNN time step error, x.shape[0]={x.shape[0]}, snn step={self.step}')
        mem = torch.zeros_like(x[0])
        u = []
        out = []
        for i in range(self.step):
            u_i, mem, out_i = mem_update(neuron_type=self.type,x_in=x[i], mem=mem, V_th=self.V_th,decay=0.25,wl=self.wl,wr=self.wr,reset=self.reset,ada=self.ada)
            u += [u_i]
            out += [out_i]
        self.u = torch.stack(u).detach()
        out = torch.stack(out)
        self.out = out.clone().detach()
        return out

class IF(nn.Module):
    def __init__(self,Vth=1, step=1, wl=0.5, wr=0.5, adaptive:str='none',reset:bool=True):
        super().__init__()
        self.step = step
        self.wl = wl
        self.wr = wr
        self.V_th = Vth
        self.reset=reset
        self.type = 'IF'
        self.ada = adaptive
        self.u = []  # before fire a spike
        self.out = []
    def forward(self, x):
        if x.dim() != 5:
            raise TypeError(f'The input dim of spiking neurons should be 5, but got the dim of {x.dim()}')
        if x.shape[0] != self.step:
            raise ValueError(f'SNN time step error')
        mem = torch.zeros_like(x[0])
        u = []
        out = []
        for i in range(self.step):
            u_i, mem, out_i = mem_update(neuron_type=self.type, x_in=x[i], mem=mem, V_th=self.V_th, decay=0.25,
                                         wl=self.wl, wr=self.wr, reset=self.reset, ada=self.ada)
            u += [u_i]
            out += [out_i]
        self.u = torch.stack(u).detach()
        out = torch.stack(out)
        self.out = out.clone().detach()
        return out
if __name__ == '__main__':
    x=torch.randn((4,4,3,32,32))
    print(IF()(x).shape)
    print(LIF()(x).shape)
