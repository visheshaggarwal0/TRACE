import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def snap_nodes_between_graphs(
    source_graph: nx.Graph, target_graph: nx.Graph, max_snap_dist: float = 10.0
) -> Dict[any, Optional[any]]:
    """Maps nodes in source_graph to the nearest node in target_graph within max_snap_dist."""
    if len(source_graph) == 0 or len(target_graph) == 0:
        return {}

    source_nodes = list(source_graph.nodes())
    target_nodes = list(target_graph.nodes())

    source_pts = np.array([source_graph.nodes[n]["pos_geo"] for n in source_nodes])
    target_pts = np.array([target_graph.nodes[n]["pos_geo"] for n in target_nodes])

    tree = cKDTree(target_pts)
    distances, indices = tree.query(source_pts, k=1)

    mapping = {}
    for i, s_node in enumerate(source_nodes):
        if distances[i] <= max_snap_dist:
            mapping[s_node] = target_nodes[indices[i]]
        else:
            mapping[s_node] = None

    return mapping


def compute_directed_apls(
    g_true: nx.Graph,
    g_prop: nx.Graph,
    sample_pairs: int = 250,
    max_snap_dist: float = 10.0,
    seed: int = 42,
) -> float:
    """Computes directed APLS from g_true to g_prop."""
    if len(g_true.nodes) < 2 or len(g_prop.nodes) < 2:
        return 0.0

    node_map = snap_nodes_between_graphs(g_true, g_prop, max_snap_dist=max_snap_dist)
    valid_true_nodes = [n for n, mapped in node_map.items() if mapped is not None]

    if len(valid_true_nodes) < 2:
        return 0.0

    rng = random.Random(seed)
    sampled_pairs = []
    attempts = 0
    max_attempts = sample_pairs * 5

    while len(sampled_pairs) < sample_pairs and attempts < max_attempts:
        attempts += 1
        s, t = rng.sample(valid_true_nodes, 2)
        if nx.has_path(g_true, s, t):
            sampled_pairs.append((s, t))

    if not sampled_pairs:
        return 0.0

    penalties = []
    for s, t in sampled_pairs:
        l_true = nx.shortest_path_length(g_true, s, t, weight="weight")
        s_prop = node_map[s]
        t_prop = node_map[t]

        if s_prop is not None and t_prop is not None and nx.has_path(g_prop, s_prop, t_prop):
            l_prop = nx.shortest_path_length(g_prop, s_prop, t_prop, weight="weight")
            diff = abs(l_true - l_prop)
            penalty = min(1.0, diff / l_true) if l_true > 0 else 0.0
        else:
            # Severed or missing connection: maximum penalty of 1.0
            penalty = 1.0

        penalties.append(penalty)

    return float(1.0 - np.mean(penalties))


def compute_apls(
    ground_truth_graph: nx.Graph,
    predicted_graph: nx.Graph,
    sample_pairs: int = 250,
    max_snap_dist: float = 10.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes Bidirectional Average Path Length Similarity (APLS).

    Returns:
        Dict[str, float]: Contains 'apls_forward', 'apls_backward', and 'apls_bidirectional'.
    """
    apls_fwd = compute_directed_apls(
        ground_truth_graph,
        predicted_graph,
        sample_pairs=sample_pairs,
        max_snap_dist=max_snap_dist,
        seed=seed,
    )
    apls_bwd = compute_directed_apls(
        predicted_graph,
        ground_truth_graph,
        sample_pairs=sample_pairs,
        max_snap_dist=max_snap_dist,
        seed=seed + 1,
    )

    if apls_fwd + apls_bwd > 0:
        apls_bidirectional = (2.0 * apls_fwd * apls_bwd) / (apls_fwd + apls_bwd)
    else:
        apls_bidirectional = 0.0

    return {
        "apls_forward": apls_fwd,
        "apls_backward": apls_bwd,
        "apls_bidirectional": apls_bidirectional,
    }
