import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.activations import ClippedReLU


class ConvBlock(nn.Module):
    def __init__(
        self, in_channel: int, out_channel: int, kernel_size: int, stride: int, activaion: int
    ):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(num_features=out_channel)
        self.act = activaion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvLayer(nn.Module):
    def __init__(
        self,
        channels: list[int],
        kernel_sizes: list[list[int, int]],
        strides: list[list[int, int]],
        activation: nn.Module,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) and len(kernel_sizes) == len(
            strides
        ), "Conv block argument size mismatch"
        channels = [1] + channels
        self.layers = nn.Sequential(
            *[
                ConvBlock(channels[i], channels[i + 1], kernel_sizes[i], strides[i], activation)
                for i in range(len(channels) - 1)
            ]
        )
        self.register_buffer("time_strides", torch.tensor(strides, dtype=torch.int32)[:, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DS2GRU(nn.Module):
    def __init__(
        self,
        nfft: int,
        conv_channels: list[int],
        conv_kernel_sizes: list[list[int, int]],
        conv_strides: list[list[int, int]],
        rnn_hidden_dim: int,
        rnn_num_layers: int,
        vocab_size: int,
        activation: nn.Module = ClippedReLU(clip_value=20),
    ):
        after_conv_nfft = nfft
        for kernel_size, stride in zip(conv_kernel_sizes, conv_strides):
            padding = kernel_size[0] // 2
            after_conv_nfft = (after_conv_nfft + 2 * padding - kernel_size[0]) // stride[0] + 1
        rnn_input_dim = conv_channels[-1] * after_conv_nfft

        super().__init__()
        self.conv_layer = ConvLayer(conv_channels, conv_kernel_sizes, conv_strides, activation)
        self.rnn = nn.GRU(rnn_input_dim, rnn_hidden_dim, rnn_num_layers, bidirectional=True, dropout=0.1)
        self.fc_classifier = nn.Linear(rnn_hidden_dim * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.activation = activation

    def transform_input_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        for i in range(self.conv_layer.time_strides.size(0)):
            input_lengths = (input_lengths - 1) // self.conv_layer.time_strides[i] + 1
        return input_lengths

    def forward(self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x (T, N, 1, H)
        x = spectrogram.permute(1, 2, 3, 0)  # x (N, 1, H, T)
        x = self.conv_layer(x)  # x (N, C, H, T)
        x = x.permute(3, 0, 1, 2).contiguous()  # (T, N, C, H)
        x = x.view(x.size(0), x.size(1), -1)  # (T, N, C * H)
        x, _ = self.rnn(x)  # (T, N, H')
        log_probs = self.log_softmax(self.activation(self.fc_classifier(x)))
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return log_probs, log_probs_length

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
