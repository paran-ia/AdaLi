from arguments import arg
import numpy as np

def log_func(wl,wr,mode):
    if arg['epoch'] == 1:
        if mode == '+':
            arg['log_kl'] = arg['min']
            arg['log_kr'] = arg['min']
            arg['log_bl'] = np.log(wl/arg['min'])/np.log(arg['Epoch'])
            arg['log_br'] = np.log(wr/arg['min']) / np.log(arg['Epoch'])
        elif mode == '-':
            arg['log_kl'] = wl
            arg['log_kr'] = wr
            arg['log_bl'] = np.log(arg['min'] / wl) / np.log(arg['Epoch'])
            arg['log_br'] = np.log(arg['min'] / wr) / np.log(arg['Epoch'])
        else:
            raise ValueError('invalid adaptive mode')
    wl = arg['log_kl']*np.exp(arg['log_bl']*np.log(arg['epoch']))
    wr = arg['log_kr']*np.exp(arg['log_br']*np.log(arg['epoch']))
    return wl,wr


def linear_func(wl,wr,mode):
    if arg['epoch'] == 1:
        if mode == '+':
            arg['linear_kl'] = (wl - arg['min']) / (arg['Epoch'] - 1)
            arg['linear_kr'] = (wr - arg['min']) / (arg['Epoch'] - 1)
            arg['linear_bl'] = arg['min'] - arg['linear_kl']
            arg['linear_br'] = arg['min'] - arg['linear_kr']
        elif mode == '-':
            arg['linear_kl'] = (arg['min'] - wl)/(arg['Epoch']-1)
            arg['linear_kr'] = (arg['min'] - wr) / (arg['Epoch'] - 1)
            arg['linear_bl'] = wl - arg['linear_kl']
            arg['linear_br'] = wr - arg['linear_kr']
        else:
            raise ValueError('invalid adaptive mode')
    wl = arg['linear_kl']*arg['epoch']+arg['linear_bl']
    wr = arg['linear_kr'] * arg['epoch'] + arg['linear_br']
    return wl,wr

def sqrt_func(wl,wr,mode):
    if arg['epoch'] == 1:
        if mode == '+':
            arg['sqrt_kl'] = (wl-arg['min']) / np.sqrt(arg['Epoch'] - 1)
            arg['sqrt_kr'] = (wr-arg['min']) / np.sqrt(arg['Epoch'] - 1)
            arg['sqrt_bl'] = arg['min'] - arg['sqrt_kl']
            arg['sqrt_br'] = arg['min'] - arg['sqrt_kr']
        if mode == '-':
            arg['sqrt_kl'] = (arg['min'] - wl) / np.sqrt(arg['Epoch'] - 1)
            arg['sqrt_kr'] = (arg['min'] - wr) / np.sqrt(arg['Epoch'] - 1)
            arg['sqrt_bl'] = wl - arg['sqrt_kl']
            arg['sqrt_br'] = wr - arg['sqrt_kr']
        else:
            raise ValueError('invalid adaptive mode')
    wl = arg['sqrt_kl'] * np.sqrt(arg['epoch']) + arg['sqrt_bl']
    wr = arg['sqrt_kr'] * np.sqrt(arg['epoch']) + arg['sqrt_br']
    return wl,wr
