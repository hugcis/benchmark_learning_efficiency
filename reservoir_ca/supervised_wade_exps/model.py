import math
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



class LinearReg(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_output,
    ):

        super().__init__()
        self.fc = nn.Linear(vocab_size, num_output)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size)
        Returns:
            out: (batch, output_size)
        """

        out = self.fc(x)
        return out


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the
        tokens in the sequence. The positional encodings have the same dimension
        as the embeddings, so that the two can be summed. Here, we use sine and
        cosine functions of different frequencies.

    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)

    """

    def __init__(self, d_model: int, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and
    a decoder."""

    def __init__(
        self,
        ntoken: int,
        ninp: int,
        nhead: int,
        nhid: int,
        nlayers: int,
        noutput: int,
        dropout: float = 0.5,
        add_dropout: bool = False,
    ):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError(
                "TransformerEncoder module does not exist in PyTorch 1.1 or lower."
            )
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.add_dropout = add_dropout
        if self.add_dropout:
            self.drop_en = nn.Dropout(p=0.6)
            self.bn2 = nn.BatchNorm1d(ninp)
        self.decoder = nn.Linear(ninp, noutput)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=False, take_mean=True):
        """ Apply the transformer.
        Args:
            src: Tensor of long, shape [padded_length, batch_size]
        """
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        emb = self.encoder(src)
        if self.add_dropout:
            emb = self.drop_en(emb)
        src = emb * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        if take_mean:
            output = output.mean(dim=0)
        if self.add_dropout:
            output = self.bn2(output)
        output = self.decoder(output)
        return output
