from dataclasses import dataclass

import torch


@dataclass
class CarlModelOutput:
    logits: torch.Tensor   # before softmax/sigmoid whatever
    activations: torch.Tensor   # after softmax/sigmoid whatever
