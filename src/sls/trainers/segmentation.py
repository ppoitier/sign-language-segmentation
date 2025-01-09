from torch import optim

from sls.trainers.base import TrainerBase
from sls.targets import get_target_decoder
from sls.backbones import MSTCNBackbone, RNNBackbone
from sls.losses import get_loss_function
from sls.metrics import PerFrameMetrics


class SegmentationTrainer(TrainerBase):
    def __init__(
            self,
            target: str,
            backbone: str,
            backbone_kwargs: dict,
            criterion: str,
            lr: float,
    ):
        super().__init__()
        self.target = target
        self.lr = lr

        if backbone == 'mstcn':
            self.backbone = MSTCNBackbone(**backbone_kwargs)
        if backbone == 'rnn':
            self.backbone = RNNBackbone(**backbone_kwargs)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.criterion = get_loss_function(criterion)

        self.decoder = get_target_decoder(target)
        self.save_hyperparameters()

        self.train_metrics = PerFrameMetrics(prefix='train/', n_classes=2)
        self.val_metrics = PerFrameMetrics(prefix='val/', n_classes=2)
        self.test_metrics = PerFrameMetrics(prefix='test/', n_classes=2)

    def prediction_step(self, batch, mode: str):
        _, features, masks, targets = batch
        train_target = targets[self.target]['train_segmentation'].long()

        logits = self.backbone(features.float(), masks)
        loss = self.criterion(logits, train_target)
        per_frame_probs = logits.softmax(dim=-1).permute(0, 2, 1).contiguous()

        # segments = self.decoder(per_frame_preds)
        # print('segments:', [s.shape for s in segments])

        val_target = targets[self.target]['val_segmentation'].long()
        batch_size = val_target.shape[0]

        if mode == 'training':
            metrics = self.train_metrics(per_frame_probs, val_target)
            self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)
        elif mode == 'validation':
            metrics = self.val_metrics(per_frame_probs, val_target)
            self.log('val_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)
        else:
            metrics = self.test_metrics(per_frame_probs, val_target)
            self.log('test_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log_metrics(metrics, batch_size=batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        return self.prediction_step(batch, mode='training')

    def validation_step(self, batch, batch_idx):
        self.prediction_step(batch, mode='validation')

    def test_step(self, batch, batch_idx):
        self.prediction_step(batch, mode='test')

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
