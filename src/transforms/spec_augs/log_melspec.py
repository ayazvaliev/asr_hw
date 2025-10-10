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

    def __call__(self, audio: torch.Tensor, audio_len: torch.Tensor, **batch) -> torch.Tensor:
        melspecs = []
        melspec_lengths = []
        for i in range(audio.size(0)):
            cur_audio_len = audio_len[i]
            cur_audio = audio[i, :, :cur_audio_len]
            log_melspec = torch.log(self.melspec_transform(cur_audio) + 1e-6)
            melspec_lengths.append(log_melspec.size(-1))
            melspecs.append(log_melspec.permute(2, 0, 1))
        melspecs = pad_sequence(melspecs, batch_first=False)
        melspec_lengths = torch.tensor(melspec_lengths, dtype=torch.int32)
        return melspecs, melspec_lengths
