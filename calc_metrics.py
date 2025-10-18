import argparse
from pathlib import Path
import os
from src.tokenizer.tokenizer_utils import normalize_text
from src.metrics.utils import calc_cer, calc_wer
from src.metrics.tracker import MetricTracker
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Calculates metrics on gt and predicted transcriptions"
)
parser.add_argument(
    "--predictions", "-p",
    type=str,
    required=True,
    help="Path to predicted transcriptions"
)
parser.add_argument(
    "--ground_truth", "-g",
    type=str,
    required=True,
    help="Path to ground truth transcriptions"
)


def main(args):
    metrics = MetricTracker("CER", "WER")
    pred_dir = Path(args.predictions)
    gt_dir = Path(args.ground_truth)
    for pred_txt in tqdm(pred_dir.rglob("*.txt"), desc="Calculating eval CER/WER metrics"):
        gt_txt = gt_dir / pred_dir.name
        if os.path.exists(gt_txt):
            with open(pred_txt, 'r') as pred_f:
                with open(gt_txt, 'r') as gt_f:
                    pred = pred_f.read().strip()
                    gt = normalize_text(gt_f.read())
                    cer = calc_cer(gt, pred)
                    wer = calc_wer(gt, pred)
                    metrics.update("CER", cer)
                    metrics.update("WER", wer)

    for key, value in metrics.result().items():
        print(f"    {key:15s}: {value}")
