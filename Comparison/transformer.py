import torch
import torch.nn as nn


class TransformerBaseline(nn.Module):
    def __init__(self, in_channels=9, d_model=88, nhead=2, num_layers=3,
                 ff_mult=3, n_classes=6, dropout=0.1, seq_len=128):
        super().__init__()
        assert d_model % nhead == 0
        self.proj = nn.Linear(in_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x) + self.pos[:, :x.size(1)]
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)
