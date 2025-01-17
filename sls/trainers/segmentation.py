import torch
from torch import optim, Tensor

from sls.trainers.base import TrainerBase
from sls.targets import get_target_decoder
from sls.backbones import MSTCNBackbone, RNNBackbone
from sls.losses import get_loss_function
from sls.metrics import PerFrameMetrics, PerSegmentMetrics


class SegmentationTrainer(TrainerBase):
    def __init__(
            self,
            encoder_name: str,
            decoder_name: str,
            decoder_args: dict,
            backbone: str,
            backbone_kwargs: dict,
            criterion: str,
            lr: float,
            use_offsets: bool,
            multi_layer_output: bool,
            criterion_weights: Tensor | None = None,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.use_offsets = use_offsets
        self.multi_layer_output = multi_layer_output
        self.lr = lr

        if 'actionness' in decoder_name:
            self.n_classes = 2
        elif 'bio_tags' in decoder_name:
            self.n_classes = 3
        else:
            raise ValueError(f"Unknown target decoder: {decoder_name}")

        self.backbone_name = backbone
        if backbone == 'mstcn':
            self.backbone = MSTCNBackbone(**backbone_kwargs)
        elif backbone == 'rnn':
            self.backbone = RNNBackbone(**backbone_kwargs)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.criterion = get_loss_function(criterion, criterion_weights)

        self.decoder = get_target_decoder(decoder_name, decoder_args)
        self.save_hyperparameters()

        self.train_metrics = PerFrameMetrics(prefix='train/', n_classes=self.n_classes)
        self.val_metrics = PerFrameMetrics(prefix='val/', n_classes=self.n_classes)
        self.test_metrics = PerFrameMetrics(prefix='test/', n_classes=self.n_classes)

        self.test_segment_metrics = PerSegmentMetrics(prefix='test/')

    def prediction_step(self, batch, mode: str):
        _, features, masks, targets = batch
        encoded_targets = targets[self.encoder_name].long()
        batch_size = encoded_targets.size(0)

        logits = self.backbone(features.float(), masks)

        loss = self.criterion(logits, encoded_targets)
        if self.use_offsets:
            loss, cls_loss, reg_loss = loss
            self.log(f'{mode}_cls_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log(f'{mode}_reg_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)

        # If we have a multi-layer output, we take the last one (the most refined one)
        if self.multi_layer_output:
            logits = logits[-1]
        # We only use classification channels for the per-frame metrics
        per_frame_probs = logits[:, :, :self.n_classes].softmax(dim=-1)
        if self.use_offsets:
            cls_targets = encoded_targets[:, :, 0]
        else:
            cls_targets = encoded_targets

        # Permute dimensions to correspond to the torchmetrics specifications
        per_frame_probs = per_frame_probs.transpose(-1, -2).contiguous()

        if mode == 'training':
            metrics = self.train_metrics(per_frame_probs, cls_targets)
        elif mode == 'validation':
            metrics = self.val_metrics(per_frame_probs, cls_targets)
        else:
            metrics = self.test_metrics(per_frame_probs, cls_targets)
        self.log_metrics(metrics, batch_size=batch_size)
        return logits, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.prediction_step(batch, mode='training')
        return loss

    def validation_step(self, batch, batch_idx):
        self.prediction_step(batch, mode='validation')

    def test_step(self, batch, batch_idx):
        _, features, masks, targets = batch
        logits, _ = self.prediction_step(batch, mode='test')
        gt_segmentation = targets[self.encoder_name].long()
        gt_segments = targets['ground_truth']['segments']
        batch_size = gt_segmentation.size(0)

        if not self.use_offsets:
            per_frame_preds = logits.argmax(dim=-1)
            pred_segments = self.decoder(per_frame_preds)
        else:
            probs_with_offsets = torch.concatenate([
                logits[:, :, :self.n_classes].softmax(dim=-1),
                logits[:, :, self.n_classes:],
            ], dim=-1)
            pred_segments = self.decoder(probs_with_offsets)

        segment_metrics = self.test_segment_metrics(pred_segments, gt_segments)
        self.log_metrics(segment_metrics, batch_size=batch_size)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
