import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

from src.tokenizer.tokenizer_utils import normalize_text
from src.metrics.utils import calc_cer, calc_wer

from pathlib import Path
import json
import pandas as pd


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
        writer=None
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
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

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
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part, rows):
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

        if metrics is not None and self.save_path is None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))
        self.log_predictions(rows=rows, **batch)

        return batch

    def log_predictions(
        self,
        rows,
        text,
        log_probs,
        log_probs_length,
        audio_path,
        **batch,
    ):
        log_probs = log_probs.detach().cpu()
        log_probs_length = log_probs_length.detach().cpu()
        argmax_inds = log_probs.argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)] for inds, ind_len in zip(argmax_inds.T, log_probs_length.numpy())
        ]
        argmax_texts = [self.text_encoder.decode(inds) for inds in argmax_inds]
        ctc_decoder_texts = self.text_encoder.ctc_decode(log_probs, log_probs_length)

        for pred, target, raw_pred, audio_path in zip(ctc_decoder_texts, text, argmax_texts, audio_path):
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
        # self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))         

    def _inference_part(self, part, dataloader, examples_to_log=50):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """
        rows = {}

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

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
                    rows=rows
                )

        # create Save dir
        if self.writer is not None:
            rows_to_log = {k: rows[k] for k in list(rows.keys())[:examples_to_log]}
            self.writer.add_table(f"predictions_{part}", pd.DataFrame.from_dict(rows_to_log, orient="index"))

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)
            with open(Path(self.save_path) / f"{part}_preds.json", "w") as f:
                json.dump(rows, f, indent=2)
            for entry in rows:
                print(entry)
                for met in self.metrics["inference"]:
                    met_entry_name = met.name[:met.name.find("_")].lower()
                    self.evaluation_metrics.update(met.name, entry[met_entry_name])
        return self.evaluation_metrics.result()
