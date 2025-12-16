from typing import Iterable, Tuple, Optional, Dict
import math, random

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def cluster_image_fast(
    edges: Iterable[Tuple],
    out_path: str = "clusters_fast.png",
    directed: bool = False,
    seed: Optional[int] = 904056181,
    layout: str = "random",          # "random" | "spring_fast"
    dpi: int = 150,
    base_node_size: int = 3,
    edge_alpha: float = 0.05,
    node_color: str = "#1f77b4",
    dedupe_edges: bool = True,
    # new:
    center_spread: bool = True,
    spread_strength: float = 1.0,    # > 0: stronger spreading of center nodes
) -> Dict:
    """
    Fast static cluster plot for medium-sized graphs (e.g. ~10–50k edges).

    Main speed tricks:
      - Smaller figure & dpi (drastically reduces rendering time).
      - Simplified layout (random or low-iteration spring).
      - Single LineCollection + single scatter call.

    Extra:
      - Optional 'center_spread' step that pushes nodes near the center
        outward more than leaves, so the middle doesn't look like a tight blob.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # --- Build graph (with optional dedupe for undirected) ---
    if not directed and dedupe_edges:
        clean = {(min(u, v), max(u, v)) for (u, v, *_) in edges}
        G = nx.Graph()
        G.add_edges_from(clean)
    else:
        G = nx.DiGraph() if directed else nx.Graph()
        for u, v, *_ in edges:
            G.add_edge(u, v)

    if G.number_of_nodes() == 0:
        raise ValueError("Empty graph.")

    # --- Layout (cheap!) ---
    if layout == "random":
        pos = nx.random_layout(G, seed=seed)
    elif layout == "spring_fast":
        pos = nx.spring_layout(
            G,
            seed=seed,
            iterations=50,
            k=None,
        )
    else:
        raise ValueError("Unknown layout (use 'random' or 'spring_fast').")

    # --- Optional: spread nodes near the center more than leaves ---
    if center_spread and G.number_of_nodes() > 1:
        nodes = list(G.nodes())
        coords = np.array([pos[n] for n in nodes], dtype=float)

        center = coords.mean(axis=0)
        rel = coords - center
        dist = np.linalg.norm(rel, axis=1)

        # Characteristic scale: average distance from center
        mean_dist = dist.mean() if dist.mean() > 0 else 1.0

        # Scale factor: ~ (1 + spread_strength) at center, ~1 far away
        # f(d) = 1 + spread_strength * exp( - (d / mean_dist)^2 )
        # so central nodes get pushed outward more than leaf nodes.
        scale = 1.0 + spread_strength * np.exp(- (dist / (mean_dist + 1e-9)) ** 2)

        rel_scaled = rel * scale[:, None]
        coords_scaled = center + rel_scaled

        # Write back into pos
        for i, n in enumerate(nodes):
            pos[n] = coords_scaled[i]

    # --- Degree-based node sizes (vectorized) ---
    deg = dict(G.degree())
    nodes = list(G.nodes())
    sizes = np.array(
        [base_node_size * math.sqrt(deg.get(n, 1)) for n in nodes],
        dtype=float,
    )

    # --- Vectorized drawing ---
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    ax.set_axis_off()

    # Edges as a single LineCollection
    if G.number_of_edges() > 0:
        segs = np.array(
            [[pos[u], pos[v]] for (u, v) in G.edges()],
            dtype=float,
        )
        lc = LineCollection(
            segs,
            colors="#0000000A",
            linewidths=0.01,
            antialiased=False,
            alpha=edge_alpha,
            capstyle="butt",
            joinstyle="miter",
        )
        ax.add_collection(lc)

    # Nodes with a single scatter call
    xy = np.array([pos[n] for n in nodes], dtype=float)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=sizes,
        c=node_color,
        edgecolors="none",
    )

    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {"graph": G, "path": out_path}
