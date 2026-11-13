from typing import List, Tuple
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def contract_graph(graph: nx.Graph) -> nx.Graph:
    """Contracts consecutive degree-2 nodes into weighted edges.

    Converts an uncontracted pixel-level graph (thousands of nodes) into a
    compact planar transportation graph where nodes represent junctions/endpoints.
    """
    contracted = graph.copy()
    degree_2_nodes = [node for node, degree in contracted.degree() if degree == 2]

    for node in degree_2_nodes:
        if node not in contracted:
            continue
        neighbors = list(contracted.neighbors(node))
        if len(neighbors) == 2:
            u, v = neighbors[0], neighbors[1]
            if u != v:
                w1 = contracted[u][node].get("weight", 1.0)
                w2 = contracted[node][v].get("weight", 1.0)
                new_weight = w1 + w2

                contracted.remove_node(node)
                contracted.add_edge(u, v, weight=new_weight)

    return contracted


def prune_short_spurs(graph: nx.Graph, min_length_meters: float = 15.0) -> nx.Graph:
    """Removes dead-end whiskers (degree-1 branches) shorter than min_length_meters."""
    pruned = graph.copy()
    changed = True

    while changed:
        changed = False
        degree_1_nodes = [n for n, d in pruned.degree() if d == 1]
        for node in degree_1_nodes:
            if node not in pruned:
                continue
            neighbors = list(pruned.neighbors(node))
            if not neighbors:
                continue
            neighbor = neighbors[0]
            edge_weight = pruned[node][neighbor].get("weight", 0.0)
            if edge_weight < min_length_meters:
                pruned.remove_node(node)
                changed = True

    return pruned


def collapse_junction_clusters(graph: nx.Graph, merge_radius_meters: float = 10.0) -> nx.Graph:
    """Collapses multi-node intersection clusters into single centroid junction nodes."""
    junction_nodes = [n for n, d in graph.degree() if d >= 3]
    if len(junction_nodes) < 2:
        return graph

    coords = np.array([graph.nodes[n]["pos_geo"] for n in junction_nodes])
    tree = cKDTree(coords)
    clusters = tree.query_ball_tree(tree, r=merge_radius_meters)

    visited = set()
    collapsed = graph.copy()

    for idx, cluster in enumerate(clusters):
        if idx in visited or len(cluster) <= 1:
            continue
        nodes_to_merge = [junction_nodes[i] for i in cluster if i not in visited]
        if len(nodes_to_merge) <= 1:
            continue

        for i in cluster:
            visited.add(i)

        # Compute centroid position
        pts = np.array([graph.nodes[n]["pos_geo"] for n in nodes_to_merge])
        centroid_geo = tuple(np.mean(pts, axis=0))
        target_node = nodes_to_merge[0]
        collapsed.nodes[target_node]["pos_geo"] = centroid_geo

        # Re-link edges to the target node
        for other_node in nodes_to_merge[1:]:
            if other_node in collapsed:
                for neighbor in list(collapsed.neighbors(other_node)):
                    if neighbor != target_node:
                        w = collapsed[other_node][neighbor].get("weight", 1.0)
                        collapsed.add_edge(target_node, neighbor, weight=w)
                collapsed.remove_node(other_node)

    return collapsed
