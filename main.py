import click
from pprint import pprint
from time import time

import torch

torch.set_float32_matmul_precision("medium")

from sls.backbones import load_module
from sls.config.utils import load_config
from sls.datasets.densely_annotated import load_datasets, load_dataloaders
from sls.datasets.utils.stats import get_class_stats
from sls.training import run_training
from sls.testing import run_testing
from sls.transforms import get_transform_pipeline


@click.command()
@click.option(
    "-c", "--config-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
def launch_experiment(config_path: str):
    config = load_config(config_path)
    print("Configuration loaded.")
    pprint(config.model_dump(mode="json"), indent=2)

    print("Loading datasets...")
    datasets = load_datasets(
        root=config.dataset.root,
        training_shards=config.dataset.train_shards_url,
        validation_shards=config.dataset.val_shards_url,
        show_progress=True,
        encoder_name=config.target.encoder.name,
        encoder_args=config.target.encoder.args,
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

    criterion_weights = None
    if config.training.criterion_use_weights:
        print("Computing class weights...")
        counts, frequencies, weights = get_class_stats(
            [
                instance_targets[config.target.encoder.name]
                for _, _, instance_targets in datasets["training"]
            ]
        )
        print("-- Class counts:", counts)
        print("-- Class frequencies:", frequencies)
        print("-- Class weights:", weights)
        criterion_weights = torch.tensor([weights[idx] for idx in range(len(weights))])

    print("Loading model...")
    module = load_module(
        encoder_name=config.target.encoder.name,
        decoder_name=config.target.decoder.name,
        decoder_args=config.target.decoder.args,
        backbone=config.backbone.name,
        backbone_kwargs=config.backbone.args,
        criterion=config.training.criterion,
        criterion_weights=criterion_weights,
        lr=config.training.learning_rate,
        use_offsets=config.target.offsets,
    )

    exp_name, exp_id = config.experiment.name, int(time() % 1e6)
    log_dir = f"{config.experiment.out_dir}/logs/{exp_name}_{exp_id}"
    print(f"Logs destination: {log_dir}")
    checkpoint_dir = f"{config.experiment.out_dir}/checkpoints/{exp_name}_{exp_id}"
    print(f"Checkpoints destination: {checkpoint_dir}")

    final_checkpoint_path = config.backbone.checkpoint_path
    if not config.training.skip_training:
        print("Launch training...")
        best_checkpoint_path = run_training(
            dataloaders,
            module,
            log_dir=log_dir,
            checkpoints_dir=checkpoint_dir,
            n_epochs=config.training.n_epochs,
            gradient_clipping=config.training.gradient_clipping,
            early_stopping_patience=config.training.early_stopping_patience,
            debug=config.experiment.debug,
        )
        if not config.experiment.debug:
            final_checkpoint_path = best_checkpoint_path

    if not config.training.skip_testing:
        print("Launch testing...")
        if final_checkpoint_path is None:
            raise ValueError("No checkpoint found for testing.")
        run_testing(
            dataloaders,
            module,
            checkpoint_path=final_checkpoint_path,
            log_dir=log_dir,
            debug=config.experiment.debug,
        )
    print("Done.")


if __name__ == "__main__":
    launch_experiment()
