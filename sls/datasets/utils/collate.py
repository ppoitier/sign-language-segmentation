import torch
from torch.utils.data import default_collate
from torch.nn.utils.rnn import pad_sequence


def collate_fixed_size(batch, target: str):
    features = default_collate([b[1] for b in batch])
    masks = default_collate([b[2] for b in batch])
    targets = {
        target: default_collate([b[3][target] for b in batch]),
        "ground_truth": {
            "segments": [
                default_collate(b[3]["ground_truth"]["segments"]) for b in batch
            ]
        },
    }
    return features, masks, targets


def collate_varying_size(batch, target: str):
    features = [torch.from_numpy(b[1]).float() for b in batch]
    lengths = [len(f) for f in features]
    features = pad_sequence(features, padding_value=-1, batch_first=True)

    masks = torch.zeros(
        features.shape[0], features.shape[1], device=features.device, dtype=torch.uint8
    )
    for idx, length in enumerate(lengths):
        masks[idx, :length] = 1

    targets = {
        target: {
            "train_segmentation": pad_sequence(
                [default_collate(b[3][target]["train_segmentation"]) for b in batch],
                padding_value=-1,
                batch_first=True,
            ),
            "val_segmentation": pad_sequence(
                [default_collate(b[3][target]["val_segmentation"]) for b in batch],
                padding_value=-1,
                batch_first=True,
            ),
        },
        "ground_truth": {
            "segments": [
                torch.from_numpy(b[3]["ground_truth"]["segments"]) for b in batch
            ],
        },
    }
    return features, masks, targets
