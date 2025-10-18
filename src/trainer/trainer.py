import os
from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.tokenizer.tokenizer_utils import normalize_text
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, batch_idx: int, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        """
        if self.is_train:
            max_num = -1
            for filename in os.listdir('.'):
                if 'spectrogram_after_intance_transforms' in filename:
                    name = filename.split('.')[0]
                    cur_num = int(name.split('_')[-1])
                    max_num = max(max_num, cur_num)
            plot_spectrogram(batch["spectrogram"][0].squeeze(0), f"spectrogram_after_batch_transforms_{max_num + 1}", save_on_disk=True)
        """

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        if self.is_train:
            mixed_precision = self.mixed_precision
        else:
            mixed_precision = torch.float32

        with torch.autocast(
            self.device_str, dtype=mixed_precision, enabled=mixed_precision is not torch.float32
        ):
            log_probs, log_probs_length = self.model(
                batch["spectrogram"], batch["spectrogram_length"]
            )
            batch.update({"log_probs": log_probs, "log_probs_length": log_probs_length})

            if self.config.DEBUG:
                print("log_probs lengths: ", batch["log_probs_length"])

            all_losses = self.criterion(**batch)
            batch.update(all_losses)
            if self.is_train:
                batch["loss"] /= self.iters_to_accumulate

        if self.is_train:
            self.grad_scaler.scale(batch["loss"]).backward()
            if ((batch_idx + 1) % self.iters_to_accumulate == 0) or (
                (batch_idx + 1) == self.epoch_len
            ):
                self.grad_scaler.unscale_(self.optimizer)
                self._clip_grad_norm()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                metrics.update("grad_norm", self._get_grad_norm())
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            saved_loss = batch[loss_name].item()
            if self.is_train:
                saved_loss *= self.iters_to_accumulate
            metrics.update(loss_name, saved_loss)

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].squeeze(0).detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot, "after_batch_transforms")
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self,
        text,
        log_probs,
        log_probs_length,
        audio_path,
        examples_to_log=10,
        **batch,
    ):
        log_probs = log_probs.detach().cpu()  # (T, N, C)
        log_probs_length = log_probs_length.detach().cpu()
        argmax_inds = log_probs.argmax(-1).numpy()  # (T, N)
        argmax_inds = [
            inds[: int(ind_len)] for inds, ind_len in zip(argmax_inds.T, log_probs_length.numpy())
        ]
        argmax_texts = [self.text_encoder.decode(inds) for inds in argmax_inds]
        ctc_decoder_texts = self.text_encoder.ctc_decode(log_probs, log_probs_length)
        tuples = list(zip(ctc_decoder_texts, text, argmax_texts, audio_path))

        rows = {}
        for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))
