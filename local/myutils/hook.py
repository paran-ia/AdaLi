import numpy as np
from functools import partial
def hook_u_spike(module,input,output,epoch,module_name):
    u = module.u.cpu().numpy()
    out = output.clone().detach().cpu().numpy()
    np.savez(f'Mp_spike/e_{epoch}_{module_name}.npz',u=u,out=out)