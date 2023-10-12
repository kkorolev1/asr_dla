import torch

from torch.optim.lr_scheduler import OneCycleLR


class ExponentialLRWithWarmup:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, max_lr, warmup):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self._rate = 0
        self.max_lr = max_lr
    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.max_lr * min(step ** (-0.5) * (self.warmup ** 0.5), step / self.warmup)

    def get_last_lr(self):
        return [self._rate]