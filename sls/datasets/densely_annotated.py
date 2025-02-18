import webdataset as wds
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sign_language_tools.annotations.transforms import SegmentationVectorToSegments
from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transform import Concatenate, Flatten
from sls.datasets.utils.collate import collate_fixed_size, collate_varying_size
from sls.datasets.utils.windows import (
    convert_instances_to_windows,
    filter_empty_windows,
)
from sls.targets import get_target_encoder


def _map_fn(
    encoder_name: str,
    encoder_args: dict,
    segment_transform,
    include_i3d_features: bool,
):
    segmentation_to_segments = SegmentationVectorToSegments(
        background_classes=(0, -1, -2), use_annotation_labels=True
    )

    def process(sample):
        binary_segmentation = sample["per_frame_binary_segmentation.npy"].astype(
            "int32"
        )
        class_segmentation = sample["per_frame_class_segmentation.npy"].astype("int32")
        segments = segmentation_to_segments(class_segmentation)
        segments = segments[:, :]

        transformed_segments = segments[:, :2].copy()
        if segment_transform is not None:
            transformed_segments = segment_transform(transformed_segments)

        encoder = get_target_encoder(
            encoder_name=encoder_name,
            encoder_args=encoder_args,
            length=binary_segmentation.shape[0],
        )
        processed_sample = {
            "features": {
                "upper_pose": sample["pose.upper_pose.npy"],
                "left_hand": sample["pose.left_hand.npy"],
                "right_hand": sample["pose.right_hand.npy"],
                "lips": sample["pose.lips.npy"],
            },
            "targets": {
                "ground_truth": {
                    "segmentation": binary_segmentation,
                    "segments": segments,
                },
                encoder_name: encoder(transformed_segments),
            },
        }
        if include_i3d_features:
            processed_sample["features"]["i3d"] = sample["i3d.npy"]
        return processed_sample

    return process


class DenselyAnnotatedSLDataset(Dataset):
    def __init__(
        self,
        url: str,
        encoder: str,
        encoder_args: dict,
        transform=None,
        segment_transform=None,
        show_progress: bool = False,
        include_i3d_features: bool = False,
        use_windows: bool = False,
        window_size: int = 1500,
        window_stride: int = 1200,
        max_empty_window_nb: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_args = encoder_args
        self.transform = transform
        self.samples: list[dict] = []
        web_dataset = wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.map(
                _map_fn(encoder, encoder_args, segment_transform, include_i3d_features)
            ),
        )
        if show_progress:
            print(f"Loading dataset [{url}].", flush=True)
        for sample in tqdm(web_dataset, disable=not show_progress, unit="samples"):
            self.samples.append(sample)

        self.use_windows = use_windows
        if use_windows:
            print("Building windows...")
            n_instances = len(self.samples)
            self.samples = convert_instances_to_windows(
                self.samples, window_size, window_stride
            )
            if max_empty_window_nb is not None:
                print("Filtering empty windows...")
                self.samples = filter_empty_windows(self.samples, max_empty_window_nb)
            print(f"From {n_instances} instances to {len(self.samples)} windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        instance_id, features, targets = (
            sample["__key__"],
            sample["features"],
            sample["targets"],
        )
        if self.transform is not None:
            features = self.transform(features)
        if self.use_windows:
            start, end = sample["start"], sample["end"]
            instance_id = f"{instance_id}_{start}_{end}"
        return instance_id, features, targets


def default_transform():
    return Compose(
        [
            Concatenate(["upper_pose", "left_hand", "right_hand", "lips"]),
            Flatten(),
        ]
    )


def load_datasets(
    root: str,
    training_shards: str,
    validation_shards: str,
    encoder_name: str,
    encoder_args: dict,
    show_progress: bool = True,
    transform=None,
    segment_transform=None,
    use_windows: bool = False,
    window_size: int = 3000,
    window_stride: int = 2800,
    max_empty_window_nb: int | None = None,
):
    return {
        "training": DenselyAnnotatedSLDataset(
            url=f"{root}/{training_shards}",
            show_progress=show_progress,
            transform=transform,
            segment_transform=segment_transform,
            encoder=encoder_name,
            encoder_args=encoder_args,
            use_windows=use_windows,
            window_size=window_size,
            window_stride=window_stride,
            max_empty_window_nb=max_empty_window_nb,
        ),
        "validation": DenselyAnnotatedSLDataset(
            url=f"{root}/{validation_shards}",
            show_progress=show_progress,
            transform=transform,
            segment_transform=segment_transform,
            encoder=encoder_name,
            encoder_args=encoder_args,
            use_windows=use_windows,
            window_size=window_size,
            window_stride=window_stride,
            max_empty_window_nb=max_empty_window_nb,
        ),
    }


def load_dataloaders(
    datasets: dict[str, DenselyAnnotatedSLDataset],
    batch_size: int,
    n_workers: int,
    fixed_sequence_length: bool = False,
):
    def _collate_fn(batch):
        instance_ids = [b[0] for b in batch]
        encoder_name = datasets["training"].encoder
        if fixed_sequence_length:
            features, masks, targets = collate_fixed_size(batch, encoder_name)
        else:
            features, masks, targets = collate_varying_size(batch, encoder_name)
        return instance_ids, features, masks, targets

    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=(x == "training"),
            num_workers=n_workers,
            collate_fn=_collate_fn,
        )
        for x in ["training", "validation"]
    }

    dataloaders["testing"] = DataLoader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=_collate_fn,
    )

    return dataloaders
