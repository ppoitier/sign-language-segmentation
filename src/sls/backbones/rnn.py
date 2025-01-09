from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        bidirectional: bool = False,
        rnn: str = 'lstm',
    ):
        super().__init__()

        if rnn == 'lstm':
            self.rnn = nn.LSTM(in_channels, hidden_channels, bidirectional=bidirectional, batch_first=True)
        elif rnn == 'gru':
            self.rnn = nn.GRU(in_channels, hidden_channels, bidirectional=bidirectional, batch_first=True)
        else:
            raise ValueError(f'Unknown RNN type: {rnn}')

        self.proj_out = nn.Linear(
            2 * hidden_channels if bidirectional else hidden_channels, out_channels
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: input of shape (N, L, C_in)
            mask: mask of shape (N, L)

        Returns:
            out of shape (N, L, C_out)
        """
        lengths = mask.sum(dim=1).cpu()
        _, total_length, _ = x.shape
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=total_length)
        x = self.proj_out(x)
        return x
