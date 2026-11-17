"""Small CPU-friendly models. Imported only when the deep-learning extra is used."""

from __future__ import annotations

import torch
from torch import nn


class ActionNetwork(nn.Module):
    def __init__(self, kind: str, n_numeric: int, n_types: int, window: int):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(n_types, 8, padding_idx=0)
        self.project = nn.Linear(n_numeric + 8, 32)
        if kind == "mlp":
            self.encoder = nn.Sequential(
                nn.Flatten(), nn.Linear(window * 32, 48), nn.ReLU(), nn.Dropout(0.1)
            )
            output_size = 48
        elif kind == "gru":
            self.encoder = nn.GRU(32, 32, batch_first=True)
            output_size = 32
        elif kind == "transformer":
            self.position = nn.Embedding(window, 32)
            layer = nn.TransformerEncoderLayer(
                32,
                nhead=4,
                dim_feedforward=64,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            output_size = 32
        else:
            raise ValueError(kind)
        self.head = nn.Linear(output_size, 2)

    def forward(self, numeric, types, lengths):
        x = torch.relu(self.project(torch.cat([numeric, self.embedding(types)], dim=-1)))
        positions = torch.arange(x.shape[1], device=x.device)
        padding = positions[None, :] >= lengths[:, None]
        x = x.masked_fill(padding[..., None], 0)
        if self.kind == "mlp":
            state = self.encoder(x)
        elif self.kind == "gru":
            sequence, _ = self.encoder(x)
            state = sequence[torch.arange(len(x), device=x.device), lengths - 1]
        else:
            x = x + self.position(positions)[None, :, :]
            # Inputs end at the present event. No future event is available.
            sequence = self.encoder(x, src_key_padding_mask=padding)
            state = sequence[torch.arange(len(x), device=x.device), lengths - 1]
        return self.head(state)
