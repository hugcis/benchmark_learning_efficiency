import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Recurrent(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_output,
        rnn_model="LSTM",
        use_last=True,
        padding_index=0,
        hidden_size=128,
        num_layers=1,
        dropout=0.0,
    ):

        super().__init__()
        self.use_last = use_last
        # embedding
        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)
        self.drop_en = nn.Dropout(p=0.6)

        # rnn module
        if rnn_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif rnn_model == "GRU":
            self.rnn = nn.GRU(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            self.rnn = nn.RNN(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        self.model = rnn_model

        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_output)

    def forward(self, x, seq_lengths) -> torch.Tensor:
        """
        Args:
            x: (batch, time_step, input_size)
        Returns:
            out: (batch, output_size)
        """

        x_embed = self.encoder(x)
        x_embed = self.drop_en(x_embed)
        packed_input = pack_padded_sequence(
            x_embed, seq_lengths.cpu().numpy(), enforce_sorted=False
        )

        # r_out shape (batch, time_step, output_size)
        packed_output, _ = self.rnn(packed_input, None)
        out_rnn, _ = pad_packed_sequence(packed_output)

        batch_indices = torch.arange(0, x.size(1)).long()

        last_tensor = out_rnn[seq_lengths - 1, batch_indices, :]

        fc_input = self.bn2(last_tensor)
        out = self.fc(fc_input)
        return out
