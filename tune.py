import warnings

import hydra
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup text_encoder
    text_encoder = instantiate(config.text_encoder)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # build model architecture, then print to console
    model = instantiate(config.model, vocab_size=len(text_encoder)).to(device)
    print(model)

    # get metrics
    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        # use text_encoder in metrics
        metrics["inference"].append(instantiate(metric_config, text_encoder=text_encoder))

    # save_path for model predictions
    if config.inferencer.get("save_path", None) is not None:
        save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
        save_path.mkdir(exist_ok=True, parents=True)
    else:
        save_path = None

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        text_encoder=text_encoder,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    def perform_search(part, init_lm_weight, init_beam_size, word_scores, lm_weights, beam_sizes):
        best_wer = 1

        best_word_score = word_scores[0]
        best_lm_weight = init_lm_weight
        best_beam_size = init_beam_size

        metric_name = "WER_(LM Guided beam search)"
        for word_score in tqdm(word_scores, desc=f"Searching for word_score for {part}"):
            text_encoder.reinitialize_decoder(
                word_score=word_score,
                lm_weight=init_lm_weight,
                beam_size=init_beam_size
            )
            cur_wer = inferencer.run_inference(part)[part][metric_name]
            if cur_wer < best_wer:
                best_wer = cur_wer
                best_word_score = word_score
                print(f'word_score: {best_word_score}, lm_weight: {init_lm_weight}, beam_size: {init_beam_size}, wer: {best_wer}')

        for lm_weight in tqdm(lm_weights, desc=f"Searching for lm_weight for {part}"):
            if lm_weight == init_lm_weight:
                continue
            text_encoder.reinitialize_decoder(
                word_score=best_word_score,
                lm_weight=lm_weight,
                beam_size=init_beam_size
            )
            cur_wer = inferencer.run_inference(part)[part][metric_name]
            if cur_wer < best_wer:
                best_wer = cur_wer
                best_lm_weight = lm_weight
                print(f'word_score: {best_word_score}, lm_weight: {best_lm_weight}, beam_size: {init_beam_size}, wer: {best_wer}')

        for beam_size in tqdm(beam_sizes, desc=f"Searching for beam_size for {part}"):
            if beam_size == init_beam_size:
                continue
            text_encoder.reinitialize_decoder(
                word_score=best_word_score,
                lm_weight=best_lm_weight,
                beam_size=beam_size
            )
            cur_wer = inferencer.run_inference(part)[part][metric_name]
            if cur_wer < best_wer:
                best_wer = cur_wer
                best_beam_size = beam_size
                print(f'word_score: {best_word_score}, lm_weight: {best_lm_weight}, beam_size: {best_beam_size}, wer: {best_wer}')

        return {
            "word_score": best_word_score,
            "lm_weight": best_lm_weight,
            "beam_size": best_beam_size,
            "wer": best_wer
        }

    parts = {
        'dev_clean': {
            'word_scores': [0],
            'init_lm_weight': 1.5,
            'init_beam_size': 200,
            'lm_weights': [],
            'beam_sizes': []
        },
        'dev_other': {
            'word_scores': [1, 2, 3],
            'init_lm_weight': 2,
            'lm_weights': [2, 3],
            'init_beam_size': 200,
            'beam_sizes': []
        }
    }
    for part, search_params in parts.items():
        tuned_params = perform_search(
            part=part,
            **search_params
        )
        print(f"Tuned params for partition {part}:")
        for name, val in tuned_params.items():
            print(f"{name} : {val}")


if __name__ == "__main__":
    main()
