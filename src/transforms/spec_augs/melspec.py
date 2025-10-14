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
        self.amplitude_to_db = AmplitudeToDB(top_db=top_db)

    def __call__(self, audio: torch.Tensor, **batch) -> torch.Tensor:
        spectrogram = self.melspec_transform(audio.squeeze(0))  # (C, H, T)
        log_spec = self.amplitude_to_db(spectrogram)
        return log_spec
