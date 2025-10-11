import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
from torch.nn.utils.rnn import pad_sequence


class LogMelSpecTransform(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        power: float,
        **kwargs,
    ):
        super().__init__()
        self.melspec_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
        )

    def __call__(self, audio: torch.Tensor, **batch) -> torch.Tensor:
        return self.melspec_transform(audio.squeeze(0)).permute(2, 0, 1)
