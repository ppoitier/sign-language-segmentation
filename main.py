import click
from pprint import pprint
from time import time

import torch
torch.set_float32_matmul_precision('medium')

from sls.backbones import load_module
from sls.config.utils import load_config
from sls.datasets.densely_annotated import load_datasets, load_dataloaders
from sls.training import run_training
from sls.transforms import get_transform_pipeline


@click.command()
@click.option(
    "-c", "--config-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
def launch_experiment(config_path: str):
    config = load_config(config_path)
    print("Configuration loaded.")
    pprint(config.model_dump(mode='json'), indent=2)

    print("Loading datasets...")
    datasets = load_datasets(
        root=config.dataset.root,
        training_shards=config.dataset.train_shards_url,
        validation_shards=config.dataset.val_shards_url,
        show_progress=True,
        target=config.target.type,
        transform=get_transform_pipeline(config.preprocessing.transforms_pipeline),
        use_windows=config.preprocessing.use_windows,
        window_size=config.preprocessing.window_size,
        window_stride=config.preprocessing.window_stride,
    )

    print("Instantiate data loaders...")
    dataloaders = load_dataloaders(
        datasets=datasets,
        batch_size=config.training.batch_size,
        n_workers=config.training.n_workers,
        fixed_sequence_length=False,
    )
    print("Data ready.")

    print("Loading model...")
    module = load_module(
        target=config.target.type,
        backbone=config.backbone.name,
        backbone_kwargs=config.backbone.args,
        criterion=config.training.criterion,
        lr=config.training.learning_rate,
    )

    exp_name, exp_id = config.experiment.name, int(time() % 1e6)
    log_dir = f"{config.experiment.out_dir}/logs/{exp_name}_{exp_id}"
    print(f"Logs destination: {log_dir}")
    checkpoint_dir = f"{config.experiment.out_dir}/checkpoints/{exp_name}_{exp_id}"
    print(f"Checkpoints destination: {checkpoint_dir}")

    print("Launch training...")
    run_training(
        dataloaders,
        module,
        log_dir=log_dir,
        checkpoints_dir=checkpoint_dir,
        n_epochs=config.training.n_epochs,
        gradient_clipping=config.training.gradient_clipping,
        debug=config.experiment.debug,
    )
    print("Done.")


if __name__ == "__main__":
    launch_experiment()
