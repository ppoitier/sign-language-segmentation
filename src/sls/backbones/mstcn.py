from torch import nn

from sls.blocks.tcn import MultiStageTCN


class MSTCNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_stages: int = 4,
        n_layers: int = 10,
    ):
        super().__init__()
        self.backbone = MultiStageTCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            n_stages=n_stages,
            n_layers=n_layers,
        )

    def forward(self, x, mask):
        return self.backbone(x.transpose(1, 2), mask.unsqueeze(1))
