import torch
import torch.nn as nn


class ClippedReLU(nn.Module):
    def __init__(self, clip_value: float):
        super().__init__()
        self.clip_value = clip_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0, max=self.clip_value)
