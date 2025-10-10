import torch
from torchaudio.transforms import Spectrogram


class LogSpecTransform:
    def __init__(self, n_fft: int, win_length: int, hop_length: int, power: float, **kwargs):
        self.spec_transform = Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=power,
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.log(self.spec_transform(audio) + 1e-6)
