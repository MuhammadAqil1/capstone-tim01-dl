
from __future__ import annotations

import torch
from torch import nn


class ModelRNN(nn.Module):

    def __init__(
        self,
        input_size: int,
        jumlah_kelas: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        jenis: str = "lstm",
        bidirectional: bool = True,
    ):
        super().__init__()

        arsitektur = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }

        jenis = jenis.lower()

        if jenis not in arsitektur:
            raise ValueError("Gunakan rnn, lstm, atau gru.")

        if input_size < 1 or jumlah_kelas < 2:
            raise ValueError("Dimensi input atau kelas tidak valid.")

        if num_layers < 1:
            raise ValueError("num_layers minimal 1.")

        self.jenis = jenis
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        arah = 2 if bidirectional else 1

        self.recurrent = arsitektur[jenis](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size * arah,
            jumlah_kelas,
        )

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                "Input harus [batch, sequence, features]."
            )

        _, hidden = self.recurrent(x)

        # LSTM mengembalikan (hidden_state, cell_state).
        if self.jenis == "lstm":
            hidden = hidden[0]

        arah = 2 if self.bidirectional else 1

        # [layers * directions, batch, hidden]
        hidden = hidden.reshape(
            self.num_layers,
            arah,
            x.shape[0],
            self.hidden_size,
        )

        # Ambil hidden state layer terakhir.
        hidden = hidden[-1]

        # Gabungkan arah forward dan backward.
        if self.bidirectional:
            hidden = torch.cat(
                [hidden[0], hidden[1]],
                dim=1,
            )

        hidden = self.dropout(hidden)

        return self.classifier(hidden)


def buat_model(
    jenis,
    input_size,
    jumlah_kelas=6,
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
    bidirectional=True,
):
    return ModelRNN(
        jenis=jenis,
        input_size=input_size,
        jumlah_kelas=jumlah_kelas,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
