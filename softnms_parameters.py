import itertools
import random

import torch
import numpy as np
from tqdm import tqdm

from sls.datasets.densely_annotated import load_datasets
from sls.transforms.utils import get_transform_pipeline, get_segment_transform_pipeline
from sls.targets.utils import get_target_decoder
from sls.metrics.segments import MeanF1ScoreOverSegments


def load_targets():
    dataset = load_datasets(
        root="/run/media/ppoitier/ppoitier/datasets/sign-languages/dgs_corpus",
        training_shards="shards_processed/shard_000000.tar",
        validation_shards="shards_processed/shard_000003.tar",
        show_progress=True,
        encoder_name="offsets+actionness",
        encoder_args=dict(),
        transform=get_transform_pipeline("norm+flatten-pose"),
        segment_transform=get_segment_transform_pipeline("none"),
        use_windows=True,
        window_size=3500,
        window_stride=2800,
    )['validation']
    targets = dict()
    for instance_id, _, instance_targets in dataset:
        targets[instance_id] = torch.from_numpy(instance_targets['ground_truth']['segments'][:, :2]).long()
    return targets


def load_results(result_filepath: str):
    results = torch.load(result_filepath, weights_only=True, map_location='cpu')
    return results


def evaluate(results, targets, soft_nms_params):
    decoder = get_target_decoder('offsets', soft_nms_params)
    metric = MeanF1ScoreOverSegments(thresholds=torch.tensor([0.5]))
    for instance_id, logits, masks in tqdm(zip(results['ids'], results['logits'], results['masks'])):
        length = masks.sum()
        logits = logits[:length]
        logits[:, :2] = logits[:, :2].softmax(dim=-1)
        gt_segments = targets[instance_id]
        pred_segments = decoder(logits.unsqueeze(0))[0]
        metric([pred_segments], [gt_segments])
    my_metrics = metric.compute()
    print(my_metrics)


if __name__ == '__main__':
    print("Loading results...")
    results = load_results("/run/media/ppoitier/ppoitier/output/sls/exp_improv2/results/dgs_pn_mstcn_4s_10l_io_ce_off_663658/results.pt")
    sample_indices = random.sample(range(len(results['ids'])), 10)
    results['ids'] = np.array(results['ids'])[sample_indices].tolist()
    results['logits'] = results['logits'][sample_indices]
    results['masks'] = results['masks'][sample_indices]

    print("Loading targets...")
    targets = load_targets()

    sigmas = np.arange(0.1, 0.91, 0.2)
    thresholds = np.arange(0.1, 0.91, 0.2)

    print("Evaluating results...")
    for sigma, threshold in itertools.product(sigmas, thresholds):
        evaluate(results, targets, dict(
            soft_nms_method='gaussian',
            soft_nms_sigma=0.5,
            soft_nms_threshold=0.5,
        ))
        break
    print("Done.")
