from torch_audiomentations import ApplyImpulseResponse
import torch.nn as nn
import torch


class ImpulseResponse(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ir_transform = ApplyImpulseResponse(**kwargs)

    def __call__(self, audio: torch.Tensor, **kwargs):
        audio = self.ir_transform(audio)
        return audio
