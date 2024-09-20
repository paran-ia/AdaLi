from config import AdaLiConfig
import numpy as np

#只有AdaLi方法需要调用此函数
def handleConfig(adaptive,epoch):
    AdaLiConfig['epoch'] = epoch
    #判断是哪种adaptive_function
    if adaptive == 'log':
        if AdaLiConfig['epoch'] == 1:
            AdaLiConfig['log_kl'] = AdaLiConfig['Left'][0]
            AdaLiConfig['log_kr'] = AdaLiConfig['Right'][0]
            AdaLiConfig['log_bl'] = np.log(AdaLiConfig['Left'][1] / AdaLiConfig['Left'][0]) / np.log(AdaLiConfig['Epoch'])
            AdaLiConfig['log_br'] = np.log(AdaLiConfig['Right'][1] / AdaLiConfig['Right'][0]) / np.log(AdaLiConfig['Epoch'])
        else:
            AdaLiConfig['Left'][0] = AdaLiConfig['log_kl'] * np.exp(AdaLiConfig['log_bl'] * np.log(AdaLiConfig['epoch']))
            AdaLiConfig['Right'][0] = AdaLiConfig['log_kr'] * np.exp(AdaLiConfig['log_br'] * np.log(AdaLiConfig['epoch']))


    elif adaptive == 'linear':
        if AdaLiConfig['epoch'] == 1:
            AdaLiConfig['linear_kl'] = (AdaLiConfig['Left'][1] - AdaLiConfig['Left'][0]) / (AdaLiConfig['Epoch'] - 1)
            AdaLiConfig['linear_kr'] = (AdaLiConfig['Right'][1] - AdaLiConfig['Right'][0]) / (AdaLiConfig['Epoch'] - 1)
            AdaLiConfig['linear_bl'] = AdaLiConfig['Left'][0] - AdaLiConfig['linear_kl']
            AdaLiConfig['linear_br'] = AdaLiConfig['Right'][0] - AdaLiConfig['linear_kr']
        else:
            AdaLiConfig['Left'][0] = AdaLiConfig['linear_kl'] * AdaLiConfig['epoch'] + AdaLiConfig['linear_bl']
            AdaLiConfig['Right'][0] = AdaLiConfig['linear_kr'] * AdaLiConfig['epoch'] + AdaLiConfig['linear_br']

    elif adaptive == 'sqrt':
        if AdaLiConfig['epoch'] == 1:
            AdaLiConfig['sqrt_kl'] = (AdaLiConfig['Left'][1] - AdaLiConfig['Left'][0]) / np.sqrt(AdaLiConfig['Epoch'] - 1)
            AdaLiConfig['sqrt_kr'] = (AdaLiConfig['Right'][1] - AdaLiConfig['Right'][0]) / np.sqrt(AdaLiConfig['Epoch'] - 1)
            AdaLiConfig['sqrt_bl'] = AdaLiConfig['Left'][0] - AdaLiConfig['sqrt_kl']
            AdaLiConfig['sqrt_br'] = AdaLiConfig['Right'][0] - AdaLiConfig['sqrt_kr']
        else:
            AdaLiConfig['Left'][0] = AdaLiConfig['sqrt_kl'] * np.sqrt(AdaLiConfig['epoch']) + AdaLiConfig['sqrt_bl']
            AdaLiConfig['Right'][0] = AdaLiConfig['sqrt_kr'] * np.sqrt(AdaLiConfig['epoch']) + AdaLiConfig['sqrt_br']
    elif adaptive == 'null':
        pass
    else:
        raise ValueError(f'{adaptive}功能暂未实现')
    print(f'AdaLiConfig["Left"][0]:{AdaLiConfig["Left"][0]}')
    print(f'AdaLiConfig["Right"][0]:{AdaLiConfig["Right"][0]}')



