from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.tokenizer.tokenizer_utils import normalize_text
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
        writer=None,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.writer = writer

        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model_ = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.evaluation_metrics = None
        if metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in metrics["inference"]],
                writer=None,
            )

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))
        self.model = torch.jit.script(self.model_)

    def run_inference(self, required_part=None):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            if required_part is not None and part != required_part:
                continue
            logs = self._inference_part(part, dataloader)
            if logs is not None:
                part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part, rows, examples_to_log=10):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        # TODO change inference logic so it suits ASR assignment
        # and task pipeline

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        log_probs, log_probs_length = self.model(batch["spectrogram"], batch["spectrogram_length"])
        batch.update({"log_probs": log_probs, "log_probs_length": log_probs_length})

        if metrics is not None:
            for met in metrics["inference"]:
                metrics.update(met.name, met(**batch))
        if self.save_path is not None:
            self._save_predictions_as_txt(**batch)

        if self.writer is not None:
            self._log_predictions(rows=rows, examples_to_log=examples_to_log, **batch)

        return batch

    def _save_predictions_as_txt(self, log_probs, log_probs_length, audio_path, **batch):
        log_probs = log_probs.detach().cpu()
        log_probs_length = log_probs_length.detach().cpu()
        ctc_decoder_texts = self.text_encoder.ctc_decode(log_probs, log_probs_length)
        for audio_path, pred in zip(audio_path, ctc_decoder_texts):
            id = Path(audio_path).stem
            with open(self.save_path / f"{id}.txt", "w") as f:
                f.write(pred + "\n")

    def _log_predictions(
        self,
        rows,
        examples_to_log,
        text,
        log_probs,
        log_probs_length,
        audio_path,
        **batch,
    ):
        if len(rows) >= examples_to_log:
            return

        log_probs = log_probs.detach().cpu()  # (T, N, C)
        log_probs_length = log_probs_length.detach().cpu()
        argmax_inds = log_probs.argmax(-1).numpy()  # (T, N)
        argmax_inds = [
            inds[: int(ind_len)] for inds, ind_len in zip(argmax_inds.T, log_probs_length.numpy())
        ]
        argmax_texts = [self.text_encoder.decode(inds) for inds in argmax_inds]
        ctc_decoder_texts = self.text_encoder.ctc_decode(log_probs, log_probs_length)
        tuples = list(zip(ctc_decoder_texts, text, argmax_texts, audio_path))

        for pred, target, raw_pred, audio_path in tuples:
            target = normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            raw_wer = calc_wer(target, raw_pred) * 100
            raw_cer = calc_cer(target, raw_pred) * 100
            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
                "raw_wer": raw_wer,
                "raw_cer": raw_cer,
            }
            if len(rows) == examples_to_log:
                return

    def _inference_part(self, part, dataloader, examples_to_log=8):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        rows = {}
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                    rows=rows,
                )

        if self.writer is not None:
            self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

        if self.evaluation_metrics is not None:
            return self.evaluation_metrics.result()
        else:
            return None
