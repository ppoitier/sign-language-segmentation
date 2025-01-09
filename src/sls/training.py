from typing import Literal

from torch.utils.data import DataLoader

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from sls.trainers.base import TrainerBase


def run_training(
    dataloaders: dict[Literal["training", "validation"], DataLoader],
    module: TrainerBase,
    log_dir: str,
    checkpoints_dir: str,
    n_epochs: int = 100,
    gradient_clipping: float = 0.0,
    debug: bool = False,
):
    checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoints_dir,
            save_top_k=1,
            every_n_epochs=10,
            save_last=True,
            monitor="val_loss",
        )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(
            logging_interval="epoch",
            log_momentum=True,
            log_weight_decay=True,
        ),
    ]
    logger = TensorBoardLogger(save_dir=log_dir)
    trainer = pl.Trainer(
        fast_dev_run=debug,
        gradient_clip_val=gradient_clipping,
        max_epochs=n_epochs,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(
        module,
        train_dataloaders=dataloaders["training"],
        val_dataloaders=dataloaders["validation"],
    )

    return checkpoint_callback.best_model_path
