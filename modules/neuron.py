from spikingjelly.activation_based.neuron import IFNode, LIFNode


class IF(IFNode):
    def __init__(self, surrogate_function=None,v_threshold =1., tau=None):
        super().__init__(v_threshold=v_threshold, v_reset=None, surrogate_function=surrogate_function, detach_reset=False, step_mode='m',
                         backend='torch', store_v_seq=False)
    def neuronal_fire(self):
        return self.surrogate_function()(self.v)

class LIF(LIFNode):
    def __init__(self,surrogate_function=None,v_threshold =1., tau=2.):
        super().__init__(tau = tau, decay_input= True, v_threshold = v_threshold,
                 v_reset = None, surrogate_function = surrogate_function,
                 detach_reset = False, step_mode='m', backend='torch', store_v_seq = False)

    def neuronal_fire(self):
        return self.surrogate_function()(self.v)

