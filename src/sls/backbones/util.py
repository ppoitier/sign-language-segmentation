from sls.trainers.segmentation import SegmentationTrainer


def load_module(
        target: str,
        backbone: str,
        backbone_kwargs: dict[str, any],
        criterion: str,
        lr: float,
):
    return SegmentationTrainer(
        target=target,
        backbone=backbone,
        backbone_kwargs=backbone_kwargs,
        criterion=criterion,
        lr=lr,
    )
