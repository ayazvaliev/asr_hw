import torch.nn as nn
import torch


class PeakNorm(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, audio: torch.Tensor):
        return audio / max(1, audio.abs().max())
