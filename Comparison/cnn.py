import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k, stride=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(cout),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CNNBaseline(nn.Module):
    def __init__(self, in_channels=9, width=92, head_hidden=120, n_classes=6, dropout=0.1):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, width, 7, dropout=dropout),
            ConvBNAct(width, width, 5, dropout=dropout),
            nn.MaxPool1d(2),
            ConvBNAct(width, width * 2, 5, dropout=dropout),
            ConvBNAct(width * 2, width * 2, 3, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(width * 2, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
