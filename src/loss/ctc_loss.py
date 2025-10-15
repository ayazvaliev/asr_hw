import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def __init__(self, blank=0, reduction="mean", zero_infinity=True):
        super().__init__(blank, reduction, zero_infinity)

    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        if torch.any(log_probs_length < text_encoded_length):
            print(f'log_probs_length < text_encoded_length: {log_probs_length}, {text_encoded_length}')
        loss = super().forward(
            log_probs=log_probs,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )

        return {"loss": loss}
