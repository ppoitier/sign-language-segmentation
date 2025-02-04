from typing import Literal
import os

import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from sls.trainers.base import TrainerBase


def save_results(results, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    torch.save(results, f"{results_dir}/results.pt")


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
