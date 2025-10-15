import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


class LogMelSpecTransform(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        power: float,
        top_db: float,
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
        assert top_db >= 0
        self.top_db = top_db
        self.amin = torch.tensor(1e-10)
        self.ref = torch.max
        # self.amplitude_to_db = AmplitudeToDB(top_db=top_db, stype="power" if power == 2.0 else "magnitude")

    def __call__(self, audio: torch.Tensor, **batch) -> torch.Tensor:
        spectrogram = self.melspec_transform(audio.squeeze(0))  # (C, H, T)
        logspec = 10 * torch.log10(torch.maximum(spectrogram, self.amin)) - 10 * torch.log10(torch.maximum(self.ref(spectrogram), self.amin))
        logspec = torch.maximum(logspec, logspec.max() - self.top_db)
        return logspec
