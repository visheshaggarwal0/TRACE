from typing import Dict, Tuple
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from affine import Affine

from ..common.geo_utils import pixel_to_geo


def extract_skeleton_graph(
    prob_map: np.ndarray,
    transform: Affine,
    prob_threshold: float = 0.5,
) -> Tuple[nx.Graph, np.ndarray]:
    """Converts a road probability map into an uncontracted 8-connected pixel Graph.

    Args:
        prob_map: 2D array of road probabilities in [0.0, 1.0].
        transform: Affine transform mapping pixel coords to geographic coordinates.
        prob_threshold: Cutoff probability to binarize roads.

    Returns:
        Tuple[nx.Graph, np.ndarray]:
            - graph: Uncontracted networkx Graph where nodes represent skeleton pixels.
            - skeleton: Binary 2D skeleton array.
    """
    binary_mask = (prob_map >= prob_threshold).astype(bool)
    skeleton = skeletonize(binary_mask)

    graph = nx.Graph()
    rows, cols = np.where(skeleton)
    pixel_set = set(zip(rows, cols))

    # 8-connected offsets: (d_row, d_col)
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for r, c in pixel_set:
        x_geo, y_geo = pixel_to_geo(c, r, transform)
        graph.add_node((r, c), pos_pixel=(c, r), pos_geo=(x_geo, y_geo))

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if (nr, nc) in pixel_set:
                # Add edge with physical Euclidean distance
                nx_geo, ny_geo = pixel_to_geo(nc, nr, transform)
                dist_meters = np.hypot(nx_geo - x_geo, ny_geo - y_geo)
                graph.add_edge((r, c), (nr, nc), weight=dist_meters)

    return graph, skeleton
