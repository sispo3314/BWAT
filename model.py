"""Block-wise adaptive temporal-scale allocation: model definition."""
 
from __future__ import annotations
 
import copy
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
@dataclass
class ModelConfig:
    in_channels: int = 9
    hidden_channels: int = 96
    n_blocks: int = 4
    kernel_sizes: Tuple[int, ...] = (5, 15, 31, 51)
    expansion: int = 2
    gate_hidden: int = 64
    stats_hidden: int = 32
    classifier_hidden: int = 128
    n_classes: int = 6
    dropout: float = 0.10
    layer_scale_init: float = 0.10
    use_input_statistics_skip: bool = True
 
    kernel_parameterization: str = "nested"
    routing_granularity: str = "blockwise"
 
    epochs: int = 80
    gumbel_tau_start: float = 3.0
    gumbel_tau_end: float = 0.30
    soft_routing_epochs: int = 10
 
    cost_weight: float = 0.01
    cost_warmup_epochs: int = 15
    cost_ramp_epochs: int = 20
 
 
class NestedDepthwiseTemporalOperator(nn.Module):
    def __init__(self, channels: int, kernel_sizes: Sequence[int]):
        super().__init__()
        kernels = tuple(int(k) for k in kernel_sizes)
        if any(k % 2 == 0 for k in kernels):
            raise ValueError("All temporal kernels must be odd.")
        self.channels = int(channels)
        self.kernel_sizes = kernels
        self.max_kernel = max(kernels)
        self.weight = nn.Parameter(
            torch.empty(channels, 1, self.max_kernel, dtype=torch.float32)
        )
        self.scale_gain = nn.Parameter(torch.ones(len(kernels), channels, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
 
    def forward_scale(self, x: torch.Tensor, scale_index: int) -> torch.Tensor:
        kernel = self.kernel_sizes[scale_index]
        offset = (self.max_kernel - kernel) // 2
        weight = self.weight[:, :, offset : offset + kernel]
        gain = self.scale_gain[scale_index].view(self.channels, 1, 1)
        return F.conv1d(
            x,
            weight * gain,
            bias=None,
            stride=1,
            padding=kernel // 2,
            groups=self.channels,
        )
 
 
class IndependentDepthwiseTemporalOperator(nn.Module):
    def __init__(self, channels: int, kernel_sizes: Sequence[int]):
        super().__init__()
        kernels = tuple(int(k) for k in kernel_sizes)
        if any(k % 2 == 0 for k in kernels):
            raise ValueError("All temporal kernels must be odd.")
        self.kernel_sizes = kernels
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=channels,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
 
    def forward_scale(self, x: torch.Tensor, scale_index: int) -> torch.Tensor:
        return self.branches[scale_index](x)
 
 
class ScaleAdaptiveBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int],
        expansion: int,
        gate_hidden: int,
        dropout: float,
        layer_scale_init: float,
        kernel_parameterization: str,
        use_local_gate: bool,
    ):
        super().__init__()
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.num_scales = len(self.kernel_sizes)
        self.use_local_gate = bool(use_local_gate)
 
        self.pre_norm = nn.BatchNorm1d(channels)
        if kernel_parameterization == "nested":
            self.temporal = NestedDepthwiseTemporalOperator(channels, kernel_sizes)
        elif kernel_parameterization == "independent":
            self.temporal = IndependentDepthwiseTemporalOperator(channels, kernel_sizes)
        else:
            raise ValueError(
                f"Unknown kernel_parameterization={kernel_parameterization}"
            )
 
        hidden = channels * expansion
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
 
        self.gate: Optional[nn.Module]
        if self.use_local_gate:
            self.gate = nn.Sequential(
                nn.Linear(channels * 2, gate_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, self.num_scales),
            )
        else:
            self.gate = None
 
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1), float(layer_scale_init))
        )
 
        max_kernel = float(max(self.kernel_sizes))
        self.register_buffer(
            "relative_costs",
            torch.tensor(
                [float(k) / max_kernel for k in self.kernel_sizes],
                dtype=torch.float32,
            ),
        )
 
    def _gate_logits(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self.gate is None:
            raise RuntimeError("Local gate requested for a global-routing block.")
        mean = x_norm.mean(dim=-1)
        std = x_norm.std(dim=-1, unbiased=False)
        return self.gate(torch.cat([mean, std], dim=1))
 
    def _all_scale_outputs(self, x_norm: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                self.temporal.forward_scale(x_norm, scale_index)
                for scale_index in range(self.num_scales)
            ],
            dim=1,
        )
 
    def _execute_selected(
        self,
        x_norm: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(x_norm)
        for scale_index in range(self.num_scales):
            mask = selected == scale_index
            if bool(mask.any()):
                output[mask] = self.temporal.forward_scale(
                    x_norm[mask], scale_index
                )
        return output
 
    def _local_probabilities(
        self,
        x_norm: torch.Tensor,
        tau: float,
        hard: bool,
        deterministic_soft: bool,
    ) -> torch.Tensor:
        logits = self._gate_logits(x_norm)
        if self.training:
            return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        if deterministic_soft:
            return torch.softmax(logits, dim=-1)
        selected = logits.argmax(dim=-1)
        return F.one_hot(selected, self.num_scales).to(x_norm.dtype)
 
    def forward(
        self,
        x: torch.Tensor,
        tau: float,
        hard: bool,
        deterministic_soft: bool = False,
        external_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = self.pre_norm(x)
        probs = (
            external_probs
            if external_probs is not None
            else self._local_probabilities(
                x_norm, tau=tau, hard=hard, deterministic_soft=deterministic_soft
            )
        )
 
        if self.training or deterministic_soft:
            all_outputs = self._all_scale_outputs(x_norm)
            temporal = (all_outputs * probs[:, :, None, None]).sum(dim=1)
        else:
            selected = probs.argmax(dim=-1)
            temporal = self._execute_selected(x_norm, selected)
 
        expected_cost = (probs * self.relative_costs[None, :]).sum(dim=-1)
        mixed = self.channel_mixer(temporal)
        output = x + self.layer_scale * mixed
        return output, probs, expected_cost
 
 
class AdaptiveTemporalScaleNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.hidden_channels
        self.cfg = cfg
        self.kernel_sizes = cfg.kernel_sizes
        self.n_blocks = cfg.n_blocks
        self.routing_granularity = cfg.routing_granularity
 
        if cfg.routing_granularity not in {"blockwise", "global"}:
            raise ValueError(
                f"Unknown routing_granularity={cfg.routing_granularity}"
            )
 
        self.stem = nn.Sequential(
            nn.Conv1d(cfg.in_channels, c, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU(),
            nn.Conv1d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU(),
        )
 
        use_local_gate = cfg.routing_granularity == "blockwise"
        self.blocks = nn.ModuleList(
            [
                ScaleAdaptiveBlock(
                    channels=c,
                    kernel_sizes=cfg.kernel_sizes,
                    expansion=cfg.expansion,
                    gate_hidden=cfg.gate_hidden,
                    dropout=cfg.dropout,
                    layer_scale_init=cfg.layer_scale_init,
                    kernel_parameterization=cfg.kernel_parameterization,
                    use_local_gate=use_local_gate,
                )
                for _ in range(cfg.n_blocks)
            ]
        )
 
        if cfg.routing_granularity == "global":
            self.global_gate: Optional[nn.Module] = nn.Sequential(
                nn.Linear(c * 2, cfg.gate_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.gate_hidden, len(cfg.kernel_sizes)),
            )
        else:
            self.global_gate = None
 
        self.final_norm = nn.BatchNorm1d(c)
        if cfg.use_input_statistics_skip:
            self.stats_encoder = nn.Sequential(
                nn.Linear(cfg.in_channels * 2, cfg.stats_hidden),
                nn.LayerNorm(cfg.stats_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            classifier_input = c * 2 + cfg.stats_hidden
        else:
            self.stats_encoder = None
            classifier_input = c * 2
 
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, cfg.classifier_hidden),
            nn.LayerNorm(cfg.classifier_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden, cfg.n_classes),
        )
 
    def _global_probabilities(
        self,
        h: torch.Tensor,
        tau: float,
        hard: bool,
        deterministic_soft: bool,
    ) -> torch.Tensor:
        if self.global_gate is None:
            raise RuntimeError("Global router is not configured.")
        pooled = torch.cat(
            [h.mean(dim=-1), h.std(dim=-1, unbiased=False)], dim=1
        )
        logits = self.global_gate(pooled)
        if self.training:
            return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        if deterministic_soft:
            return torch.softmax(logits, dim=-1)
        selected = logits.argmax(dim=-1)
        return F.one_hot(selected, len(self.kernel_sizes)).to(h.dtype)
 
    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        hard: bool = True,
        deterministic_soft: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, in_channels, T) -> logits (B, n_classes),
        routing weights (B, n_blocks, n_scales), block costs (B, n_blocks)."""
        input_mean = x.mean(dim=-1)
        input_std = x.std(dim=-1, unbiased=False)
 
        h = self.stem(x)
        global_probs: Optional[torch.Tensor] = None
        if self.routing_granularity == "global":
            global_probs = self._global_probabilities(
                h,
                tau=tau,
                hard=hard,
                deterministic_soft=deterministic_soft,
            )
 
        route_probs: List[torch.Tensor] = []
        block_costs: List[torch.Tensor] = []
        for block in self.blocks:
            h, probs, expected_cost = block(
                h,
                tau=tau,
                hard=hard,
                deterministic_soft=deterministic_soft,
                external_probs=global_probs,
            )
            route_probs.append(probs)
            block_costs.append(expected_cost)
 
        h = self.final_norm(h)
        temporal_summary = torch.cat(
            [h.mean(dim=-1), h.amax(dim=-1)], dim=1
        )
        if self.stats_encoder is not None:
            stats = self.stats_encoder(torch.cat([input_mean, input_std], dim=1))
            final_feature = torch.cat([temporal_summary, stats], dim=1)
        else:
            final_feature = temporal_summary
 
        logits = self.classifier(final_feature)
        routes = torch.stack(route_probs, dim=1)
        costs = torch.stack(block_costs, dim=1)
        return logits, routes, costs
 
 
def tau_at_epoch(epoch: int, cfg: ModelConfig) -> float:
    if cfg.epochs <= 1:
        return cfg.gumbel_tau_end
    progress = (epoch - 1) / (cfg.epochs - 1)
    ratio = cfg.gumbel_tau_end / cfg.gumbel_tau_start
    return float(cfg.gumbel_tau_start * (ratio ** progress))
 
 
def cost_weight_at_epoch(epoch: int, cfg: ModelConfig) -> float:
    if epoch <= cfg.cost_warmup_epochs:
        return 0.0
    progress = (epoch - cfg.cost_warmup_epochs) / max(cfg.cost_ramp_epochs, 1)
    return float(cfg.cost_weight * min(max(progress, 0.0), 1.0))
 
 
def adaptive_scale_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    costs: torch.Tensor,
    cost_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(logits, targets)
    observed_cost = costs.mean()
    return ce + cost_weight * observed_cost, ce, observed_cost
 
 
def make_ablation_config(base_cfg: ModelConfig, mode: str) -> ModelConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.cost_weight = base_cfg.cost_weight
    cfg.kernel_parameterization = "nested"
    cfg.use_input_statistics_skip = True
    cfg.routing_granularity = "blockwise"
 
    if mode == "full_model":
        pass
    elif mode == "no_cost_regularization":
        cfg.cost_weight = 0.0
    elif mode == "independent_kernels":
        cfg.kernel_parameterization = "independent"
    elif mode == "global_single_route":
        cfg.routing_granularity = "global"
    else:
        raise ValueError(f"Unknown ablation mode={mode}")
    return cfg
 
 
if __name__ == "__main__":
    cfg = ModelConfig()
    model = AdaptiveTemporalScaleNet(cfg)
    x = torch.randn(8, cfg.in_channels, 128)
 
    model.eval()
    with torch.no_grad():
        logits, routes, costs = model(x, tau=0.1, hard=True)
 
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
 
    print(f"logits          : {tuple(logits.shape)}")
    print(f"routes          : {tuple(routes.shape)}")
    print(f"costs           : {tuple(costs.shape)}")
    print(f"trainable params: {n_params:,}")
    print(f"selected scales : {routes.argmax(dim=-1)[0].tolist()}")
 

