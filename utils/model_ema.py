import copy

import torch


class ModelEma:
    """Minimal exponential moving average wrapper used by EchoMask."""

    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for name, value in self.ema.state_dict().items():
            incoming = source[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(incoming, alpha=1.0 - self.decay)
            else:
                value.copy_(incoming)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        return self.ema.load_state_dict(state_dict)
