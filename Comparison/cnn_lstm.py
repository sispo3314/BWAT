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


class CNNLSTMBaseline(nn.Module):
    def __init__(self, in_channels=9, conv_width=72, lstm_hidden=124,
                 lstm_layers=1, n_classes=6, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_channels, conv_width, 7, dropout=dropout),
            ConvBNAct(conv_width, conv_width, 5, dropout=dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_width,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(dropout if lstm_layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, lstm_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_classes),
        )

    def forward(self, x):
        x = self.conv(x).transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.head(x)
