import torch


class WarmupScheduler:
    def __init__(self, *args, **kwargs):
        self.warmup_steps = kwargs["warmup_steps"]
        self.training_steps = kwargs["training_steps"]
        del kwargs["warmup_steps"]
        del kwargs["training_steps"]

        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step / self.warmup_steps)
            else:
                return max(0.0, float(self.training_steps - current_step) / float(max(1, self.training_steps - self.warmup_steps)))

        kwargs.update({"lr_lambda": lr_lambda})
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(*args, **kwargs)


    def state_dict(self):
        return self.scheduler.state_dict()


    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


    def step(self):
        self.scheduler.step()


    def get_last_lr(self):
        return self.scheduler.get_last_lr()


if __name__ == "__main__":
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0042)
    warmup_scheduler = WarmupScheduler(optimizer=optimizer, warmup_steps=100, training_steps=5000)
    lrs = []
    for i in range(5000):
        optimizer.step()
        warmup_scheduler.step()
        lrs.append(warmup_scheduler.get_last_lr())
    print(lrs)