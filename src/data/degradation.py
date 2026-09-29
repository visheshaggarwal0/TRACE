import random
from typing import Tuple
import geopandas as gpd
from shapely.geometry import Point, box


def random_edge_dropout(
    gdf: gpd.GeoDataFrame, drop_ratio: float = 0.30, seed: int = 42
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Randomly drops a fraction of road segments to simulate an incomplete basemap.

    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
            - prior_gdf: The degraded map (e.g. 70% remaining).
            - residual_gdf: The missing roads that the model must discover (e.g. 30%).
    """
    if gdf.empty:
        return gdf.copy(), gdf.copy()

    rng = random.Random(seed)
    total_edges = len(gdf)
    indices = list(range(total_edges))
    rng.shuffle(indices)

    drop_count = int(total_edges * drop_ratio)
    dropped_indices = set(indices[:drop_count])
    kept_indices = [i for i in range(total_edges) if i not in dropped_indices]

    prior_gdf = gdf.iloc[kept_indices].copy().reset_index(drop=True)
    residual_gdf = gdf.iloc[list(dropped_indices)].copy().reset_index(drop=True)

    return prior_gdf, residual_gdf


def suburban_expansion_mask(
    gdf: gpd.GeoDataFrame, radius_fraction: float = 0.65
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Simulates rapid suburban expansion by removing road segments outside a central radius."""
    if gdf.empty:
        return gdf.copy(), gdf.copy()

    total_bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center_x = (total_bounds[0] + total_bounds[2]) / 2.0
    center_y = (total_bounds[1] + total_bounds[3]) / 2.0
    max_extent = max(total_bounds[2] - total_bounds[0], total_bounds[3] - total_bounds[1])
    radius = (max_extent / 2.0) * radius_fraction

    center_point = Point(center_x, center_y)
    central_buffer = center_point.buffer(radius)

    # Roads that intersect the central buffer are kept; outer roads are considered residual/new
    is_inside = gdf.geometry.intersects(central_buffer)
    prior_gdf = gdf[is_inside].copy().reset_index(drop=True)
    residual_gdf = gdf[~is_inside].copy().reset_index(drop=True)

    return prior_gdf, residual_gdf


def disaster_severance_mask(
    gdf: gpd.GeoDataFrame, bbox_fraction: float = 0.25
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Simulates localized infrastructure destruction (e.g. flood/landslide) within a localized zone."""
    if gdf.empty:
        return gdf.copy(), gdf.copy()

    total_bounds = gdf.total_bounds
    width = total_bounds[2] - total_bounds[0]
    height = total_bounds[3] - total_bounds[1]

    sub_w = width * bbox_fraction
    sub_h = height * bbox_fraction

    # Place localized disaster zone near the center
    minx = total_bounds[0] + width * 0.35
    miny = total_bounds[1] + height * 0.35
    maxx = minx + sub_w
    maxy = miny + sub_h

    disaster_box = box(minx, miny, maxx, maxy)

    is_severed = gdf.geometry.intersects(disaster_box)
    prior_gdf = gdf[~is_severed].copy().reset_index(drop=True)
    residual_gdf = gdf[is_severed].copy().reset_index(drop=True)

    return prior_gdf, residual_gdf


def simulate_incomplete_map(
    gdf: gpd.GeoDataFrame, strategy: str = "random_drop", **kwargs
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Unified dispatcher for simulating incomplete prior basemaps."""
    if strategy == "random_drop":
        return random_edge_dropout(gdf, **kwargs)
    elif strategy == "suburban_ring":
        return suburban_expansion_mask(gdf, **kwargs)
    elif strategy == "disaster_bbox":
        return disaster_severance_mask(gdf, **kwargs)
    else:
        raise ValueError(f"Unknown degradation strategy: {strategy}")
