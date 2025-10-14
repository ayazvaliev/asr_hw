import warnings

import hydra
import torch
from hydra.utils import instantiate, get_method, get_class, call
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

import os

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config, resolve=True)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    if os.path.exists(config.tokenizer.tokenizer.model_path):
        tokenizer = instantiate(config.tokenizer.tokenizer)
    else:
        train_tokenizer = get_method(config.tokenizer.tokenizer_trainer.method)
        tokenizer = train_tokenizer(**project_config["tokenizer"]["tokenizer_trainer"]["train_args"])

    # setup text_encoder
    text_encoder = instantiate(config.text_encoder,
                               logger=logger,
                               tokenizer=tokenizer)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # build model architecture, then print to console
    model = instantiate(config.model, vocab_size=tokenizer.get_vocab_size())
    logger.info(model)

    # instantiate train and inference metrics
    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            # use text_encoder in metrics
            metrics[metric_type].append(instantiate(metric_config, text_encoder=text_encoder))

    trainer = Trainer(
        model=model,
        metrics=metrics,
        text_encoder=text_encoder,
        config=config,
        project_config=project_config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms
    )

    trainer.train()


if __name__ == "__main__":
    main()
