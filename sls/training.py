from typing import Literal

from torch.utils.data import DataLoader

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from sls.trainers.base import TrainerBase


def run_training(
    dataloaders: dict[Literal["training", "validation"], DataLoader],
    module: TrainerBase,
    log_dir: str,
    checkpoints_dir: str,
    n_epochs: int = 100,
    gradient_clipping: float = 0.0,
    early_stopping_patience: int = 10,
    debug: bool = False,
):
    checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoints_dir,
            save_top_k=1,
            save_last=True,
            monitor="validation_loss",
        )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(
            logging_interval="epoch",
            log_momentum=True,
            log_weight_decay=True,
        ),
        EarlyStopping(
            monitor="validation_loss",
            patience=early_stopping_patience,
        )
    ]
    tb_logger = TensorBoardLogger(name="tb", save_dir=log_dir)
    csv_logger = CSVLogger(name="csv", save_dir=log_dir)
    trainer = pl.Trainer(
        fast_dev_run=debug,
        gradient_clip_val=gradient_clipping,
        max_epochs=n_epochs,
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
    )
    trainer.fit(
        module,
        train_dataloaders=dataloaders["training"],
        val_dataloaders=dataloaders["validation"],
    )

    return checkpoint_callback.best_model_path
