from typing import Literal

from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from sls.trainers.base import TrainerBase


def run_testing(
    dataloaders: dict[Literal["testing"], DataLoader],
    module: TrainerBase,
    checkpoint_path: str,
    log_dir: str,
    debug: bool = False,
):
    logger = TensorBoardLogger(save_dir=log_dir)
    trainer = pl.Trainer(
        fast_dev_run=debug,
        logger=logger,
    )
    trainer.test(
        module,
        ckpt_path=checkpoint_path,
        dataloaders=dataloaders["testing"],
    )
