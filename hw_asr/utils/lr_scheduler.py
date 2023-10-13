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
    

class TransformerLrScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, multiplier=5):
        self._optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.n_steps = 0
        self.multiplier = multiplier
        self._lr = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict) 

    def step(self):
        self.n_steps += 1
        self._lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self._lr

    def _get_lr(self):
        return self.multiplier * (self.d_model ** -0.5) * min(self.n_steps ** (-0.5), self.n_steps * (self.warmup_steps ** (-1.5)))
  
    def get_last_lr(self):
        return [self._lr]