"""
model.py — NeuralCascade GNN Architecture
==========================================
Implements a three-component graph neural network for predicting prion
propagation cascades over protein–protein interaction (PPI) networks:

    1. Graph Attention Network (GAT) encoder
       — 8 attention heads × 128 dims/head = aggregated protein context
    2. GRU-based Temporal Message Passing
       — T=6 timesteps mirror the six clinical/pathological stages of
         neurodegenerative disease progression (e.g., Braak staging in AD,
         or Hoehn & Yahr scale in PD). Each GRU step propagates conformational
         templating from seed proteins to neighbours, simulating the prion-like
         spread mechanism.
    3. MLP Risk-Scoring Head
       — Maps temporal hidden state → scalar risk score in [0, 1] via sigmoid.

Loss:
    BCE (binary classification of prion-prone vs. normal) combined with
    a differentiable Kendall-Tau ranking term (SPEARMAN approx via pairwise)
    so that the model learns not just whether a protein misfolds but *when*
    in the cascade it does so.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import get_laplacian
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Hyper-parameters for the NeuralCascade GNN.

    Attributes:
        node_feat_dim: Dimensionality of input node features (ESM-2 1280 +
            PLAAC 1 + IUPred2A 1 + GTEx brain expression 3 + Simplex 1 + 
            Boltz-1 1 = 1287).
        gat_heads: Number of attention heads in the GATv2 layer.
        gat_dim_per_head: Output dimension per attention head.
        diffusion_hidden_dim: Hidden state size of the continuous diffusion.
        diffusion_time: Total diffusion time horizon.
        diffusion_steps: Number of Euler integration steps.
        mlp_dims: MLP hidden layer widths for the risk-scoring head.
        dropout: Dropout probability applied after activation layers.
    """
    node_feat_dim: int = 1287
    gat_heads: int = 8
    gat_dim_per_head: int = 128
    diffusion_hidden_dim: int = 256
    diffusion_time: float = 6.0       # T=6 ← corresponds to Braak stages
    diffusion_steps: int = 12         # finer integration steps
    mlp_dims: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2


# ---------------------------------------------------------------------------
# Component 1 — Multi-head Graph Attention Network encoder
# ---------------------------------------------------------------------------

class GATEncoder(nn.Module):
    """Two-layer GATv2 encoder.

    Unlike standard GAT, GATv2 allows every node to attend to every other
    node in a fully dynamic manner (dynamic attention). This captures
    complex heterotypic interactions during templating.

    Args:
        in_channels: Dimension of raw input node features.
        heads: Number of parallel attention heads.
        out_per_head: Output dimension per head (total = heads × out_per_head).
        dropout: Dropout rate applied to attention coefficients.
    """

    def __init__(
        self,
        in_channels: int,
        heads: int = 8,
        out_per_head: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_per_head,
            heads=heads,
            dropout=dropout,
            concat=True,           # concatenate heads → heads × out_per_head
        )
        # Second layer takes heads×out_per_head as input
        self.gat2 = GATv2Conv(
            in_channels=heads * out_per_head,
            out_channels=out_per_head,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.bn1 = nn.BatchNorm1d(heads * out_per_head)
        self.bn2 = nn.BatchNorm1d(heads * out_per_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Encode node features via two-layer multi-head GAT.

        Args:
            x: Node feature matrix [N, in_channels].
            edge_index: Edge connectivity [2, E].

        Returns:
            Encoded node embeddings [N, heads × out_per_head].
        """
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        return x  # shape: [N, 8 × 128] = [N, 1024]


# ---------------------------------------------------------------------------
# Component 2 — Continuous Spread Modeling (Diffusion Neural ODE)
# ---------------------------------------------------------------------------

class ContinuousDiffusion(nn.Module):
    """Neural ODE inspired continuous diffusion of misfolding signals.

    Replaces discrete GRU steps with continuous spread modeling. The process
    is governed by a diffusion-like differential equation:
        dh(t)/dt = MLP_diffuse(A_norm @ h(t)) - decay * h(t)
    
    This mathematically bounds the cascade as a continuous wavefront over the
    topology, integrated using Euler steps.

    Args:
        input_dim: Dimension of per-node GATv2 embeddings.
        hidden_dim: Diffusion representation space dimension.
        time: Total diffusion time horizon (e.g., T=6.0 for Braak staging).
        steps: Number of integration steps (dt = time/steps).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        time: float = 6.0,
        steps: int = 12,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time = time
        self.steps = steps
        self.dt = time / steps

        # Non-linear diffusion operator (learns propagation rate)
        self.diffusivity = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()  # Tanh bounds the derivative
        )

        # Decay parameter (protein homeostasis / clearance)
        self.clearance = nn.Parameter(torch.ones(1) * 0.1)

        # Linear projection: GAT output → Initial condition h(0)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

    def forward(
        self,
        gat_out: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Propagate misfolding signals via continuous PDE numerical integration.

        Args:
            gat_out: GAT node embeddings [N, input_dim].
            edge_index: Edge connectivity [2, E].

        Returns:
            Final hidden state after diffusion time horizon [N, hidden_dim].
        """
        N = gat_out.size(0)
        
        # Initial condition h(0)
        h = torch.tanh(self.input_proj(gat_out))  # [N, hidden_dim]

        src, dst = edge_index

        # Simple Euler integration of: dh/dt = Diffuse(Neighbours) - clearance * h
        for _ in range(self.steps):
            # 1. Aggregate neighbours (Laplacian-like smoothing)
            agg = torch.zeros(N, self.hidden_dim, device=gat_out.device)
            agg.scatter_add_(
                0,
                dst.unsqueeze(1).expand(-1, self.hidden_dim),
                h[src],
            )
            # Degree normalisation
            degree = torch.zeros(N, device=gat_out.device)
            degree.scatter_add_(0, dst, torch.ones(src.size(0), device=gat_out.device))
            degree = degree.clamp(min=1).unsqueeze(1)
            agg = agg / degree

            # 2. Compute derivative dh/dt
            # The signal flowing in is transformed by the diffusivity network,
            # and the current signal decays via clearance.
            dh_dt = self.diffusivity(agg) - (torch.relu(self.clearance) * h)

            # 3. Step forward in time: h(t + dt) = h(t) + dh/dt * dt
            h = h + dh_dt * self.dt

        return h  # shape: [N, hidden_dim=256]


# ---------------------------------------------------------------------------
# Component 3 — MLP Risk-Scoring Head
# ---------------------------------------------------------------------------

class RiskScoringHead(nn.Module):
    """MLP that maps temporal hidden states to a prion-spread risk score.

    Architecture: 256 → 128 → 64 → 1 with sigmoid output.
    The sigmoid constrains output to [0, 1], interpretable as the probability
    that a given protein will be recruited into the misfolding cascade.

    Args:
        hidden_dim: Dimension of incoming temporal embeddings (256).
        mlp_dims: Sequence of hidden layer widths.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        mlp_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = hidden_dim
        for out_dim in mlp_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))  # final projection to scalar
        self.net = nn.Sequential(*layers)

    def forward(self, h: Tensor) -> Tensor:
        """Compute per-node risk score.

        Args:
            h: Temporal hidden states [N, hidden_dim].

        Returns:
            Risk scores in [0, 1], shape [N, 1].
        """
        return torch.sigmoid(self.net(h))  # [N, 1]


# ---------------------------------------------------------------------------
# Full NeuralCascade Model
# ---------------------------------------------------------------------------

class NeuralCascade(nn.Module):
    """Full GNN model stacking GATv2 → ContinuousDiffusion → RiskHead.

    Args:
        config: ModelConfig dataclass with all hyper-parameters.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        gat_out_dim = config.gat_heads * config.gat_dim_per_head  # 1024

        self.encoder = GATEncoder(
            in_channels=config.node_feat_dim,
            heads=config.gat_heads,
            out_per_head=config.gat_dim_per_head,
            dropout=config.dropout,
        )
        self.temporal_mp = ContinuousDiffusion(
            input_dim=gat_out_dim,
            hidden_dim=config.diffusion_hidden_dim,
            time=config.diffusion_time,
            steps=config.diffusion_steps,
        )
        self.risk_head = RiskScoringHead(
            hidden_dim=config.diffusion_hidden_dim,
            mlp_dims=config.mlp_dims,
            dropout=config.dropout,
        )

    def forward(self, data: Data) -> Tensor:
        """Full forward pass over a PyG Data object.

        Args:
            data: PyTorch Geometric Data with .x [N, 1287],
                  .edge_index [2, E], and optionally .edge_attr.

        Returns:
            Per-node risk scores [N, 1] in [0, 1].
        """
        x, edge_index = data.x, data.edge_index

        # Step 1: Encode protein context via dynamic graph attention
        gat_emb = self.encoder(x, edge_index)        # [N, 1024]

        # Step 2: Propagate misfolding signals via continuous diffusion PDE
        h_final = self.temporal_mp(gat_emb, edge_index)  # [N, 256]

        # Step 3: Score each protein's recruitment risk
        risk = self.risk_head(h_final)                # [N, 1]
        return risk


# ---------------------------------------------------------------------------
# Combined Loss Function
# ---------------------------------------------------------------------------

def neural_cascade_loss(
    pred: Tensor,
    target: Tensor,
    alpha: float = 0.7,
    tau_weight: float = 0.3,
) -> Tensor:
    """Combined BCE + soft Kendall-Tau ranking loss.

    Rationale:
        Binary cross-entropy alone teaches whether a protein misfolds.
        The ranking term forces the model to also learn *cascade order* —
        critically important for identifying intervention targets upstream of
        the propagation front. Kendall-Tau is approximated by counting
        concordant vs. discordant pairs in the predicted score ordering
        versus the ground-truth label ordering.

    Args:
        pred: Predicted risk scores [N, 1] or [N].
        target: Binary ground-truth labels [N] (1 = prion-prone).
        alpha: Weight on BCE loss (1−alpha on Kendall-Tau term).
        tau_weight: Scale factor for the ranking term.

    Returns:
        Scalar combined loss.
    """
    pred = pred.squeeze(-1)   # ensure shape [N]
    target = target.float()

    # --- Binary cross-entropy term ---
    bce = F.binary_cross_entropy(pred, target)

    # --- Soft pairwise Kendall-Tau approximation ---
    # For every pair (i, j), compute sign(target_i − target_j) and check
    # whether pred_i − pred_j agrees in sign (concordant pair).
    # We use a sigmoid-smoothed version so gradients flow through.
    N = pred.size(0)
    if N < 2:
        return bce  # can't compute pairwise with a single node

    # Pairwise difference matrices
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)    # [N, N]
    target_diff = target.unsqueeze(0) - target.unsqueeze(1)  # [N, N]

    # Soft concordance: sigmoid of pred_diff aligned with sign(target_diff)
    concordance = torch.sigmoid(pred_diff * target_diff * 10.0)

    # Mask diagonal (self-pairs) and compute mean concordance
    mask = ~torch.eye(N, dtype=torch.bool, device=pred.device)
    soft_tau = concordance[mask].mean()

    # We want to *maximise* concordance → minimise (1 − concordance)
    ranking_loss = 1.0 - soft_tau

    return alpha * bce + tau_weight * ranking_loss


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_model(config: Optional[ModelConfig] = None) -> NeuralCascade:
    """Construct a NeuralCascade model with optional config override.

    Sets global seeds for reproducibility before model construction.

    Args:
        config: Optional ModelConfig; uses defaults if None.

    Returns:
        Initialised NeuralCascade model.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    return NeuralCascade(config=config)
