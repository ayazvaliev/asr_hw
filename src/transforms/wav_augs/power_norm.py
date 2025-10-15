import torch.nn as nn
import torch


class PowerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        rms = audio.pow(2).mean().sqrt()
        return audio / rms
