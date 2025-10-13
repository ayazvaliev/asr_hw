import warnings

import hydra
import torch
from hydra.utils import instantiate, get_method, get_class, call
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

# from src.model import DS2, BaselineModel
from src.trainer.trainer_utils import get_optimizer_grouped_parameters

from math import ceil
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
    model = instantiate(config.model, vocab_size=tokenizer.get_vocab_size()).to(device)
    logger.info(model)
    model = torch.jit.script(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            # use text_encoder in metrics
            metrics[metric_type].append(instantiate(metric_config, text_encoder=text_encoder))

    optimizer_cls = get_class(config.optimizer.cls)
    grouped_trainable_params = get_optimizer_grouped_parameters(model, config.optimizer.weight_decay)
    optimizer = optimizer_cls(grouped_trainable_params, **project_config["optimizer"]["optimizer_config"])

    if config.trainer.gradient_accumulation is None:
        gradient_accumulation = config.dataloader.batch_size
    else:
        gradient_accumulation = config.trainer.gradient_accumulation

    accumulate_iters = gradient_accumulation // config.dataloader.batch_size
    total_steps = ceil(len(dataloaders["train"]) // accumulate_iters) * config.trainer.n_epochs
    lr_scheduler = call(config.lr_scheduler, optimizer=optimizer, num_warmup_steps=total_steps * config.lr_scheduler_config.warmup_ratio, num_training_steps=total_steps)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        gradient_accumulation=gradient_accumulation
    )

    trainer.train()


if __name__ == "__main__":
    main()
