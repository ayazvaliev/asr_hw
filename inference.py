import os
import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

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

    if config.writer.enabled:
        writer = instantiate(config.writer)
    else:
        writer = None

    # setup text_encoder
    text_encoder = instantiate(config.text_encoder)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # build model architecture, then print to console
    model = instantiate(config.model, vocab_size=len(text_encoder))
    print(model)

    # get metrics
    if config.get("metrics", None) is not None:
        metrics = {"inference": []}
        for metric_config in config.metrics.get("inference", []):
            # use text_encoder in metrics
            metrics["inference"].append(instantiate(metric_config, text_encoder=text_encoder))
        if len(metrics["inference"]) == 0:
            metrics = None
    else:
        metrics = None

    # save_path for model predictions
    save_path = config.inferencer.get("save_path", None)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)

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
        writer=writer,
    )

    res_metrics = inferencer.run_inference()
    if save_path is not None:
        print(f"All predictions saved in {save_path.absolute().resolve()}")
    for key, value in res_metrics.items():
        print(f"    {key:15s}: {value}")


if __name__ == "__main__":
    main()
