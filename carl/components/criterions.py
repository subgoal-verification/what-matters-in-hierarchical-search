from abc import ABC, abstractmethod

import torch
from torch import nn

from carl.components.modeling_output import CarlModelOutput


class CarlCriterion(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, output: CarlModelOutput, *args, **kwargs) -> torch.Tensor():
        pass


class CriterionWithLogits(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, output: CarlModelOutput, *args, **kwargs) -> torch.Tensor():
        return self.loss(output.logits, *args, **kwargs)


class CriterionWithActivations(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, output: CarlModelOutput, *args, **kwargs) -> torch.Tensor():
        return self.loss(output.activations, *args, **kwargs)
