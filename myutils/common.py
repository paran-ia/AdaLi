import torch
from config import AdaLiConfig

def heaviside(x: torch.Tensor):
    return (x >= AdaLiConfig['Vth']).to(x)

def cal_firing_rate(s_seq: torch.Tensor):
    # s_seq.shape = [T, N, *]
    fr = s_seq.flatten(0).mean(0)
    return fr

def percentage_in_range(tensor, lower_bound, upper_bound):
    # 统计在区间范围内的值的数量
    in_range_count = ((tensor >= lower_bound) & (tensor <= upper_bound)).sum().item()
    # 总数量
    total_count = tensor.numel()
    # 计算占比
    percentage = in_range_count / total_count * 100
    return round(percentage, 2)

def cal_mp_summary_statistics(mp_list):
    min_list=[]
    q1_list = []
    mean_list = []
    q3_list = []
    max_list =[]
    compute_density=[]
    for mp in mp_list:
        mp.detach().cpu()
        min_list.append(torch.min(mp))
        q1_list.append(torch.quantile(mp,0.25))
        mean_list.append(torch.mean(mp))
        q3_list.append(torch.quantile(mp, 0.75))
        max_list.append(torch.max(mp))
        compute_density.append(percentage_in_range(mp,AdaLiConfig['Vth']-AdaLiConfig['Left'][0],AdaLiConfig['Vth']+AdaLiConfig['Right'][0]))
    mean_min = sum(min_list)/len(min_list)
    mean_q1 = sum(q1_list)/len(q1_list)
    mean = sum(mean_list)/len(mean_list)
    mean_q3 = sum(q3_list) / len(q3_list)
    mean_max = sum(max_list) / len(max_list)
    compute_density = sum(compute_density)/len(compute_density)
    return mean_min,mean_q1,mean,mean_q3,mean_max,compute_density





