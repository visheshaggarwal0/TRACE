import numpy as np
import cv2
from typing import List, Tuple, Union
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from affine import Affine
from ..common.geo_utils import geo_to_pixel


def rasterize_linestrings(
    geometries: Union[gpd.GeoDataFrame, List[Union[LineString, MultiLineString]]],
    shape: Tuple[int, int],
    transform: Affine,
    line_thickness_pixels: int = 4,
    gaussian_sigma: float = 0.0,
) -> np.ndarray:
    """Burns vector LineString geometries into a 2D float32 raster mask in [0, 1].

    Args:
        geometries: GeoDataFrame or list of Shapely LineStrings / MultiLineStrings.
        shape: Output raster shape as (height, width).
        transform: Affine transform from pixel space to geographic coordinates.
        line_thickness_pixels: Pixel thickness for drawing road centerlines.
        gaussian_sigma: Optional Gaussian smoothing sigma for spatial prior channel.

    Returns:
        np.ndarray: 2D float32 array with shape (height, width) with values in [0.0, 1.0].
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)

    lines: List[LineString] = []
    if isinstance(geometries, gpd.GeoDataFrame):
        for geom in geometries.geometry:
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, LineString):
                lines.append(geom)
            elif isinstance(geom, MultiLineString):
                lines.extend(geom.geoms)
    else:
        for geom in geometries:
            if isinstance(geom, LineString):
                lines.append(geom)
            elif isinstance(geom, MultiLineString):
                lines.extend(geom.geoms)

    for line in lines:
        coords = list(line.coords)
        pixel_points = []
        for x_geo, y_geo in coords:
            col, row = geo_to_pixel(x_geo, y_geo, transform)
            pixel_points.append([int(round(col)), int(round(row))])

        if len(pixel_points) >= 2:
            pts = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=line_thickness_pixels)

    mask_float = mask.astype(np.float32)

    if gaussian_sigma > 0.0:
        kernel_size = int(math_ceil_odd(gaussian_sigma * 4))
        mask_float = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), gaussian_sigma)
        # Normalize to maximum of 1.0
        max_val = np.max(mask_float)
        if max_val > 0:
            mask_float = mask_float / max_val

    return mask_float


def math_ceil_odd(val: float) -> int:
    """Returns the smallest odd integer greater than or equal to val."""
    c = int(np.ceil(val))
    return c if c % 2 == 1 else c + 1
