import math
from typing import Tuple, Union
import geopandas as gpd
from pyproj import CRS
from affine import Affine
from shapely.geometry import Point, LineString


def get_utm_crs_for_latlon(lat: float, lon: float) -> CRS:
    """Computes the appropriate UTM CRS (WGS 84 / UTM zone XXN or XXS) for a given lat/lon."""
    zone = int(math.floor((lon + 180) / 6) + 1)
    hemisphere = "north" if lat >= 0 else "south"
    epsg_code = 32600 + zone if hemisphere == "north" else 32700 + zone
    return CRS.from_epsg(epsg_code)


def reproject_geodataframe(gdf: gpd.GeoDataFrame, target_crs: Union[str, CRS, int]) -> gpd.GeoDataFrame:
    """Safely reprojects a GeoDataFrame to the specified target CRS."""
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS defined.")
    return gdf.to_crs(target_crs)


def geo_to_pixel(x_geo: float, y_geo: float, transform: Affine) -> Tuple[float, float]:
    """Converts geographic/projected map coordinates (X, Y) to pixel coordinates (col, row).

    Uses the inverse affine transformation of the raster image.
    """
    inv_transform = ~transform
    col, row = inv_transform @ (x_geo, y_geo)
    return col, row


def pixel_to_geo(col: float, row: float, transform: Affine) -> Tuple[float, float]:
    """Converts image pixel coordinates (col, row) to geographic/projected map coordinates (X, Y)."""
    x_geo, y_geo = transform @ (col, row)
    return x_geo, y_geo


def snap_point_to_linestring(point: Point, line: LineString) -> Point:
    """Finds the nearest point along a LineString to a target Point."""
    distance_along_line = line.project(point)
    return line.interpolate(distance_along_line)
