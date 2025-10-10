from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class WERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self,
        log_probs: Tensor,
        log_probs_length: Tensor,
        text: List[str],
        **kwargs,
    ):
        wers = []
        predictions = self.text_encoder.ctc_decode(log_probs, log_probs_length)
        for pred_text, target_text in zip(predictions, text):
            target_text = BaseMetric.normalize_text(target_text)
            wer = calc_wer(target_text, pred_text)
            if wer is not None:
                wers.append(wer)
        return sum(wers) / len(wers)
