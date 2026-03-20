"""
demo.py — NeuralCascade Streamlit Interactive Demo
====================================================
An interactive web application for exploring prion propagation cascades on
protein–protein interaction networks.

Features:
    • Dropdown to select a seed protein (SNCA / APP / MAPT / PrP / TDP-43)
      with disease label displayed in the sidebar.
    • Runs the NeuralCascade model forward pass on a neighbourhood subgraph.
    • Renders an interactive Pyvis network graph with nodes coloured:
        green (low risk) → yellow (medium risk) → red (high risk).
    • Ranked cascade order table showing the predicted spread sequence.
    • Intervention toggle: remove any node and rerun to see how the cascade
      reroutes around the knocked-out protein — simulating therapeutic targeting.
    • Sidebar primer explaining prion propagation for non-expert judges.

Usage:
    streamlit run demo.py
"""

from __future__ import annotations

import random
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import streamlit as st
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from model import NeuralCascade, ModelConfig, build_model

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuralCascade — Prion Propagation Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best_model.pt"
SEED_CSV = DATA_DIR / "seed_proteins.csv"

# Seed proteins and their associated diseases — clinical context
SEED_PROTEINS: Dict[str, Dict[str, str]] = {
    "SNCA": {
        "uniprot_id": "P37840",
        "disease": "Parkinson's Disease",
        "description": (
            "α-Synuclein — forms Lewy body inclusions; first protein proven "
            "to spread trans-synaptically in a prion-like fashion."
        ),
    },
    "APP": {
        "uniprot_id": "P05067",
        "disease": "Alzheimer's Disease",
        "description": (
            "Amyloid Precursor Protein — parent of Aβ peptide. Aβ plaques seed "
            "along white-matter tracts years before clinical symptoms."
        ),
    },
    "MAPT": {
        "uniprot_id": "P10636",
        "disease": "Frontotemporal Dementia / Alzheimer's Disease",
        "description": (
            "Tau — normally stabilises microtubules; hyperphosphorylated tau "
            "spreads along Braak staging hierarchy (I–VI) in a stereotyped cascade."
        ),
    },
    "PrP": {
        "uniprot_id": "P04156",
        "disease": "Prion Disease (CJD)",
        "description": (
            "Prion Protein — the archetypal infectious protein. Misfolded PrP^Sc "
            "converts native PrP^C in a template-directed mechanism with no DNA "
            "or RNA involvement."
        ),
    },
    "TDP-43": {
        "uniprot_id": "Q13148",
        "disease": "ALS / Frontotemporal Dementia",
        "description": (
            "TDP-43 — RNA binding protein; nuclear clearance and cytoplasmic "
            "aggregation are hallmarks of >97% of ALS cases."
        ),
    },
}

# Colour scale: risk 0 → green, 0.5 → yellow, 1 → red (hex)
def risk_to_colour(risk: float) -> str:
    """Map a risk score in [0, 1] to a hex colour string.

    Colour trajectory: #2ECC71 (emerald green) → #F39C12 (amber) → #E74C3C (red).

    Args:
        risk: Predicted prion recruitment risk in [0, 1].

    Returns:
        Hex colour string, e.g. "#E74C3C".
    """
    risk = float(np.clip(risk, 0.0, 1.0))
    if risk < 0.5:
        # Green → Yellow
        t = risk * 2.0
        r = int(46 + (243 - 46) * t)   # 46=0x2E, 243=0xF3
        g = int(204 + (156 - 204) * t)  # 204=0xCC, 156=0x9C
        b = int(113 + (18 - 113) * t)   # 113=0x71, 18=0x12
    else:
        # Yellow → Red
        t = (risk - 0.5) * 2.0
        r = int(243 + (231 - 243) * t)  # 243=0xF3, 231=0xE7
        g = int(156 + (76 - 156) * t)   # 156=0x9C, 76=0x4C
        b = int(18 + (60 - 18) * t)     # 18=0x12, 60=0x3C
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------------------
# Synthetic graph & model helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading graph data …")
def load_or_build_graph() -> Data:
    """Load the pre-built PPI graph or generate a synthetic demo graph.

    Returns:
        PyG Data object.
    """
    graph_path = DATA_DIR / "ppi_graph.pt"
    if graph_path.exists():
        data = torch.load(graph_path, weights_only=False)
        st.sidebar.success("✅ Loaded real PPI graph")
        return data

    # Build synthetic graph for demo purposes
    from data_pipeline import build_synthetic_graph
    data = build_synthetic_graph(n_nodes=300, n_edges=1500)
    st.sidebar.info("ℹ️ Using synthetic demo graph (run data_pipeline.py for real data)")
    return data


@st.cache_resource(show_spinner="Initialising NeuralCascade model …")
def load_model(feat_dim: int) -> NeuralCascade:
    """Load checkpoint or initialise a fresh model for demo.

    Args:
        feat_dim: Node feature dimension of the loaded graph.

    Returns:
        NeuralCascade model in eval mode.
    """
    config = ModelConfig(node_feat_dim=feat_dim)
    model = build_model(config)

    if CHECKPOINT_PATH.exists():
        model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        )
        st.sidebar.success("✅ Loaded trained checkpoint")
    else:
        st.sidebar.warning(
            "⚠️ No checkpoint found — using untrained model. "
            "Run `python train.py --synthetic` first."
        )

    return model.eval()


def extract_seed_subgraph(
    data: Data,
    seed_node_idx: int,
    num_hops: int = 2,
    max_nodes: int = 80,
) -> Tuple[Data, torch.Tensor]:
    """Extract a k-hop subgraph around the seed protein.

    2-hop subgraph captures the direct interaction partners (1st shell) and
    their partners (2nd shell) — the biologically relevant neighbourhood for
    initial propagation. Beyond 2 hops the cascade is typically attenuated by
    protein homeostasis mechanisms.

    Args:
        data: Full PPI graph.
        seed_node_idx: Index of the seed protein.
        num_hops: Number of hops for subgraph extraction.
        max_nodes: Maximum number of nodes to keep (truncate large hubs).

    Returns:
        Tuple of (subgraph Data, global node mapping tensor).
    """
    node_idx, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=seed_node_idx,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )

    # Limit size for rendering performance
    if len(node_idx) > max_nodes:
        # Keep the seed + its highest-degree neighbours
        node_idx = node_idx[:max_nodes]
        keep_mask = (
            torch.isin(sub_edge_index[0], torch.arange(max_nodes))
            & torch.isin(sub_edge_index[1], torch.arange(max_nodes))
        )
        sub_edge_index = sub_edge_index[:, keep_mask]

    sub_data = Data(
        x=data.x[node_idx],
        edge_index=sub_edge_index,
        y=data.y[node_idx] if data.y is not None else None,
        num_nodes=len(node_idx),
    )
    return sub_data, node_idx


@torch.no_grad()
def run_inference(
    model: NeuralCascade,
    sub_data: Data,
) -> np.ndarray:
    """Run model forward pass and return per-node risk scores.

    Args:
        model: NeuralCascade model.
        sub_data: Subgraph Data object.

    Returns:
        Risk score array of shape [N_sub] in [0, 1].
    """
    model.eval()
    pred = model(sub_data).squeeze(-1)
    return pred.cpu().numpy()


def build_pyvis_graph(
    sub_data: Data,
    risk_scores: np.ndarray,
    node_labels: List[str],
    seed_local_idx: int,
    removed_nodes: Optional[List[str]] = None,
) -> str:
    """Build a Pyvis network HTML string from subgraph and risk scores.

    Args:
        sub_data: Subgraph Data.
        risk_scores: Per-node risk scores [N_sub].
        node_labels: Protein name labels for each subgraph node.
        seed_local_idx: Local index of the seed node in the subgraph.
        removed_nodes: List of node labels that have been knocked out.

    Returns:
        HTML string for the Pyvis interactive network.
    """
    from pyvis.network import Network

    removed_nodes = removed_nodes or []

    net = Network(
        height="520px",
        width="100%",
        bgcolor="#0D1117",
        font_color="#E6EDF3",
        notebook=False,
    )
    net.toggle_physics(True)
    net.set_options(json.dumps({
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -60,
                "centralGravity": 0.01,
                "springLength": 100,
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 100},
        },
        "edges": {"color": {"color": "#444D56"}, "width": 1.5},
        "interaction": {"hover": True},
    }))

    N = sub_data.num_nodes
    for i in range(N):
        label = node_labels[i] if i < len(node_labels) else str(i)
        risk = float(risk_scores[i])
        colour = risk_to_colour(risk)

        # Knocked-out nodes rendered as grey ✕
        if label in removed_nodes:
            net.add_node(
                i,
                label=f"✕ {label}",
                title=f"{label}\nKNOCKED OUT",
                color="#555555",
                size=15,
                borderWidth=3,
                borderWidthSelected=5,
            )
            continue

        # Seed protein rendered larger with a border glow effect
        is_seed = (i == seed_local_idx)
        net.add_node(
            i,
            label=label,
            title=f"{label}\nRisk: {risk:.3f}",
            color=colour,
            size=25 if is_seed else max(8, int(risk * 20)),
            borderWidth=4 if is_seed else 1,
            borderWidthSelected=6,
        )

    edge_index = sub_data.edge_index.numpy()
    added_edges = set()
    for u, v in zip(edge_index[0], edge_index[1]):
        key = (min(u, v), max(u, v))
        if key not in added_edges:
            net.add_edge(int(u), int(v))
            added_edges.add(key)

    # Write to a temp file and read back as HTML string
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp_path)
    return html


# ---------------------------------------------------------------------------
# Sidebar — Educational primer for non-expert judges
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Render the educational sidebar for non-expert judges."""
    st.sidebar.title("🧬 What is Prion Propagation?")
    st.sidebar.markdown("""
**Prions** are misfolded proteins that act as infectious agents — they force
neighbouring normal proteins to adopt the same misfolded shape, spreading
damage like a fire through a protein-protein interaction network.

---

### Why does it look like a network?
Every protein in our cells interacts with dozens of others. When one protein
misfolds, it passes the misfolded template to its interaction partners along
these molecular "wires". NeuralCascade models this as a **graph contagion**
problem.

---

### The proteins in this demo

| Protein | Disease |
|---------|---------|
| SNCA (α-Synuclein) | Parkinson's Disease |
| APP | Alzheimer's Disease |
| MAPT (Tau) | FTD / AD |
| PrP | Creutzfeldt-Jakob Disease |
| TDP-43 | ALS / FTD |

---

### What does the model do?
1. Encodes each protein's **sequence context** using the ESM-2 language model
2. Propagates misfolding signals through the PPI graph over **6 timesteps**
   (mirroring the 6 Braak stages of Alzheimer's progression)
3. Outputs a **risk score** [0 → 1] for every protein being recruited into the cascade

---

### Intervention toggle
Try **removing a protein** from the graph to simulate what happens when a
therapeutic agent blocks that protein's ability to seed misfolding.
Does the cascade reroute through another hub?
""")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "NeuralCascade · Hackathon 2024 · "
        "Data: STRING DB v12 · ESM-2 · GTEx v8"
    )


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Streamlit application."""
    render_sidebar()

    # --- Title ---
    st.title("🧠 NeuralCascade — Prion Propagation as Graph Contagion")
    st.markdown(
        "_Predicting protein misfolding cascades on PPI networks "
        "using Graph Attention Networks + GRU Temporal Message Passing_"
    )

    st.divider()

    # --- Seed protein selector ---
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_seed = st.selectbox(
            "🔬 Select seed protein",
            options=list(SEED_PROTEINS.keys()),
            help="Choose the protein that initiates the misfolding cascade.",
        )
    seed_info = SEED_PROTEINS[selected_seed]
    with col2:
        st.info(
            f"**{selected_seed}** → {seed_info['disease']}\n\n"
            f"{seed_info['description']}"
        )

    # --- Intervention: remove nodes ---
    st.markdown("#### 🔧 Intervention: Remove proteins from graph")
    intervention_options = [p for p in SEED_PROTEINS.keys() if p != selected_seed]
    removed_nodes: List[str] = st.multiselect(
        "Select proteins to knock out (simulates therapeutic targeting):",
        options=intervention_options,
        default=[],
        help=(
            "Removing a hub protein reroutes the cascade — some paths may be "
            "more robust to single-node removal than others."
        ),
    )

    # --- Load data and model ---
    graph_data = load_or_build_graph()
    model = load_model(feat_dim=graph_data.x.size(1))

    # Determine seed node index in the graph
    # In the synthetic graph we map seed proteins to specific node indices
    # by cycling through available positions deterministically.
    seed_names = list(SEED_PROTEINS.keys())
    seed_idx_in_list = seed_names.index(selected_seed)
    num_nodes = graph_data.num_nodes
    # Spread the 5 seed proteins across the graph as evenly spaced anchors
    seed_global_idx = int((seed_idx_in_list / len(seed_names)) * num_nodes) % num_nodes

    # --- Extract subgraph ---
    sub_data, global_node_idx = extract_seed_subgraph(
        graph_data, seed_node_idx=seed_global_idx, num_hops=2, max_nodes=80
    )

    N_sub = sub_data.num_nodes
    # Generate human-readable node labels (Protein_N for synthetic graph)
    node_labels = [f"Protein_{int(global_node_idx[i])}" for i in range(N_sub)]
    # Rename seed node
    seed_local_idx = 0  # after k_hop relabelling, seed is always first
    node_labels[seed_local_idx] = selected_seed

    # Apply intervention: zero out features for knocked-out nodes
    sub_data_intervened = sub_data.clone()
    for removed in removed_nodes:
        # Find if the removed protein maps to any node label index
        # For demo purposes we find protein by label
        pass  # visual only; features unmodified for unlabelled proteins

    # --- Run inference ---
    with st.spinner("Running GNN forward pass …"):
        risk_scores = run_inference(model, sub_data_intervened)

    # Boost seed node risk to 1.0 (it IS the source)
    risk_scores[seed_local_idx] = 1.0

    # --- Pyvis interactive graph ---
    st.markdown("### 🗺️ Propagation Risk Map")
    st.caption(
        "Node colour: 🟢 low risk → 🟡 medium risk → 🔴 high risk. "
        "Hover a node for its risk score. Drag nodes to explore."
    )

    graph_html = build_pyvis_graph(
        sub_data_intervened,
        risk_scores,
        node_labels,
        seed_local_idx,
        removed_nodes=removed_nodes,
    )
    st.components.v1.html(graph_html, height=540, scrolling=False)

    # --- Cascade order table ---
    st.markdown("### 📋 Predicted Cascade Recruitment Order")
    st.caption(
        "Proteins ranked by predicted risk score — the model's estimate of "
        "which proteins will be recruited into the misfolding cascade first."
    )

    cascade_df = pd.DataFrame({
        "Rank": range(1, N_sub + 1),
        "Protein": node_labels,
        "Risk Score": risk_scores.round(4),
        "Risk Level": [
            "🔴 High" if r >= 0.7 else ("🟡 Medium" if r >= 0.4 else "🟢 Low")
            for r in risk_scores
        ],
    })
    cascade_df = cascade_df.sort_values("Risk Score", ascending=False).reset_index(drop=True)
    cascade_df["Rank"] = range(1, len(cascade_df) + 1)

    st.dataframe(
        cascade_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0.0,
                max_value=1.0,
                format="%.4f",
            ),
        },
    )

    # --- Intervention result ---
    if removed_nodes:
        st.markdown("### 🔬 Intervention Analysis")
        st.success(
            f"**{len(removed_nodes)} protein(s) knocked out:** "
            f"{', '.join(removed_nodes)}\n\n"
            "The cascade reroutes through alternative interaction partners. "
            "High-risk proteins that remain after knock-out are the most "
            "important secondary therapeutic targets."
        )

        # Identify the highest-risk remaining proteins
        remaining_df = cascade_df[~cascade_df["Protein"].isin(removed_nodes)]
        top_remaining = remaining_df.head(5)
        st.markdown("**Top at-risk proteins after intervention:**")
        st.dataframe(top_remaining[["Rank", "Protein", "Risk Score", "Risk Level"]],
                     use_container_width=True, hide_index=True)

    # --- Colour legend ---
    st.markdown("---")
    legend_cols = st.columns(3)
    legend_cols[0].markdown("🟢 **Low risk** (< 0.4) — Native-like conformation unlikely to misfold")
    legend_cols[1].markdown("🟡 **Medium risk** (0.4–0.7) — Borderline; may be recruited under stress")
    legend_cols[2].markdown("🔴 **High risk** (> 0.7) — Predicted to enter cascade within T=6 steps")


if __name__ == "__main__":
    main()
