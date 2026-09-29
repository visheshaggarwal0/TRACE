import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString, mapping


def compute_heading_angle_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Computes the azimuth/heading angle in degrees from p1 to p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx)) % 360.0


def snap_subgraph_to_master(
    new_graph: nx.Graph,
    master_graph: nx.Graph,
    search_radius_meters: float = 30.0,
    max_angle_tolerance_deg: float = 40.0,
) -> nx.Graph:
    """Topologically stitches and snaps a newly extracted road subgraph into the master graph.

    For each degree-1 boundary terminal in new_graph:
        1. Calculates the road's approach tangent vector.
        2. Searches within search_radius_meters for candidate edges in master_graph.
        3. Projects the terminal point onto the candidate edge.
        4. Splits the master edge and inserts a new degree-3 or degree-4 junction.
    """
    unified = nx.compose(master_graph, new_graph)
    terminal_nodes = [n for n, d in new_graph.degree() if d == 1]

    for terminal in terminal_nodes:
        if terminal not in unified:
            continue
        term_pos = np.array(unified.nodes[terminal]["pos_geo"])
        neighbors = list(new_graph.neighbors(terminal))
        if not neighbors:
            continue
        prev_node = neighbors[0]
        prev_pos = np.array(new_graph.nodes[prev_node]["pos_geo"])

        # Road direction vector
        direction = term_pos - prev_pos
        norm = np.linalg.norm(direction)
        if norm < 1e-5:
            continue
        unit_dir = direction / norm

        # Search for closest master edge
        best_candidate = None
        min_proj_dist = float("inf")

        for u, v, data in master_graph.edges(data=True):
            p_u = np.array(master_graph.nodes[u]["pos_geo"])
            p_v = np.array(master_graph.nodes[v]["pos_geo"])
            edge_vec = p_v - p_u
            edge_len_sq = np.dot(edge_vec, edge_vec)
            if edge_len_sq < 1e-5:
                continue

            # Project terminal point onto edge segment [p_u, p_v]
            t = np.clip(np.dot(term_pos - p_u, edge_vec) / edge_len_sq, 0.0, 1.0)
            proj_point = p_u + t * edge_vec
            dist = np.linalg.norm(term_pos - proj_point)

            if dist < search_radius_meters and dist < min_proj_dist:
                min_proj_dist = dist
                best_candidate = (u, v, proj_point, dist)

        if best_candidate is not None:
            u, v, proj_point, snap_dist = best_candidate
            new_junction_id = f"snap_junc_{terminal[0]}_{terminal[1]}"
            proj_geo = tuple(proj_point)

            # Insert junction node on master edge
            unified.add_node(new_junction_id, pos_geo=proj_geo, pos_pixel=unified.nodes[terminal].get("pos_pixel"))

            # Re-split master edge if it exists
            if unified.has_edge(u, v):
                w_orig = unified[u][v].get("weight", np.linalg.norm(np.array(master_graph.nodes[u]["pos_geo"]) - np.array(master_graph.nodes[v]["pos_geo"])))
                dist_u_proj = np.linalg.norm(np.array(master_graph.nodes[u]["pos_geo"]) - proj_point)
                dist_proj_v = np.linalg.norm(proj_point - np.array(master_graph.nodes[v]["pos_geo"]))

                unified.remove_edge(u, v)
                unified.add_edge(u, new_junction_id, weight=dist_u_proj)
                unified.add_edge(new_junction_id, v, weight=dist_proj_v)

            # Connect new road terminal to the newly created junction
            unified.add_edge(terminal, new_junction_id, weight=snap_dist)

    return unified


def export_graph_to_geojson(graph: nx.Graph, output_geojson_path: str, crs: str = "EPSG:32611") -> gpd.GeoDataFrame:
    """Exports a NetworkX road graph with pos_geo coordinates to a standard GeoJSON file."""
    geometries: List[LineString] = []
    weights: List[float] = []

    for u, v, data in graph.edges(data=True):
        if "pos_geo" in graph.nodes[u] and "pos_geo" in graph.nodes[v]:
            p1 = graph.nodes[u]["pos_geo"]
            p2 = graph.nodes[v]["pos_geo"]
            line = LineString([p1, p2])
            geometries.append(line)
            weights.append(data.get("weight", line.length))

    gdf = gpd.GeoDataFrame(
        {"weight_meters": weights, "geometry": geometries},
        crs=crs,
    )
    gdf.to_file(output_geojson_path, driver="GeoJSON")
    return gdf
