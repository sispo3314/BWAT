import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=8, expand=2, conv_kernel=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.dwconv = nn.Conv1d(
            self.d_inner, self.d_inner, conv_kernel,
            groups=self.d_inner, padding=conv_kernel - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))[None, :].repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        u, z = self.in_proj(x).chunk(2, dim=-1)

        u = self.dwconv(u.transpose(1, 2))[..., :x.size(1)].transpose(1, 2)
        u = F.silu(u)

        dt_raw, B_t, C_t = torch.split(self.x_proj(u), [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw)) + 1e-4
        A = -torch.exp(self.A_log)

        batch, length, _ = u.shape
        state = torch.zeros(batch, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(length):
            dt_t = dt[:, t, :]
            u_t = u[:, t, :]
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB_u = dt_t.unsqueeze(-1) * B_t[:, t, :].unsqueeze(1) * u_t.unsqueeze(-1)
            state = dA * state + dB_u
            ys.append((state * C_t[:, t, :].unsqueeze(1)).sum(-1) + self.D.unsqueeze(0) * u_t)
        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        y = self.dropout(self.out_proj(y))
        return residual + y


class MambaBaseline(nn.Module):
    def __init__(self, in_channels=9, d_model=96, num_layers=4, d_state=8,
                 expand=2, n_classes=6, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList([
            SelectiveSSMBlock(d_model, d_state=d_state, expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        x = self.proj(x.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)
