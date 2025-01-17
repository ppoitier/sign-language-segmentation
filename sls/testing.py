from typing import Literal

from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from sls.trainers.base import TrainerBase


def run_testing(
    dataloaders: dict[Literal["testing"], DataLoader],
    module: TrainerBase,
    checkpoint_path: str,
    log_dir: str,
    debug: bool = False,
):
    tb_logger = TensorBoardLogger(name="tb", save_dir=log_dir)
    csv_logger = CSVLogger(name="csv", save_dir=log_dir)
    trainer = pl.Trainer(
        fast_dev_run=debug,
        logger=[tb_logger, csv_logger],
    )
    trainer.test(
        module,
        ckpt_path=checkpoint_path,
        dataloaders=dataloaders["testing"],
    )
