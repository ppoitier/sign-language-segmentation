import torch
from torch import Tensor

from sls.trainers.segmentation import SegmentationTrainer


def load_module(
        encoder_name: str,
        decoder_name: str,
        decoder_args: dict,
        backbone: str,
        backbone_kwargs: dict[str, any],
        criterion: str,
        lr: float,
        use_offsets: bool,
        multilayer_output: bool,
        criterion_weights: Tensor | None = None,
):
    return SegmentationTrainer(
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        decoder_args=decoder_args,
        backbone=backbone,
        backbone_kwargs=backbone_kwargs,
        criterion=criterion,
        lr=lr,
        use_offsets=use_offsets,
        criterion_weights=criterion_weights,
        multi_layer_output=multilayer_output,
    )


def load_module_from_checkpoint(checkpoint_path: str):
    map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return SegmentationTrainer.load_from_checkpoint(checkpoint_path, map_location=map_location)
