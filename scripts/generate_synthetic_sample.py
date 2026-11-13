import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import LineString
import cv2


def generate_synthetic_tiles(output_dir: str = "data/raw", num_tiles: int = 5, tile_size: int = 512):
    """Generates synthetic satellite GeoTIFF tiles and matching GeoJSON road networks

    for local development, end-to-end debugging, and pipeline testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_tiles} synthetic GeoTIFF tiles and GeoJSON road networks in {output_dir}...")

    base_x = 665000.0
    base_y = 4000000.0
    pixel_size = 0.5  # 0.5m GSD

    for i in range(num_tiles):
        tile_id = f"synthetic_tile_{i+1:03d}"
        origin_x = base_x + (i * tile_size * pixel_size * 1.5)
        origin_y = base_y
        transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

        # 1. Background terrain (dirt, sand, vegetation noise)
        rgb = np.random.normal(loc=110, scale=20, size=(tile_size, tile_size, 3)).astype(np.float32)
        rgb = np.clip(rgb, 50, 180).astype(np.uint8)
        img = np.ascontiguousarray(rgb)

        # 2. Road layout
        lines_geo = []

        # Main Highway across tile
        y_hwy_pixel = 256 + int(np.random.uniform(-40, 40))
        hwy_pts = [(0, y_hwy_pixel), (tile_size - 1, y_hwy_pixel)]
        cv2.line(
            img,
            hwy_pts[0],
            hwy_pts[1],
            color=(70, 70, 75),  # Dark asphalt
            thickness=12,
        )

        p1_geo = (origin_x + 0 * pixel_size, origin_y - y_hwy_pixel * pixel_size)
        p2_geo = (origin_x + (tile_size - 1) * pixel_size, origin_y - y_hwy_pixel * pixel_size)
        lines_geo.append(LineString([p1_geo, p2_geo]))

        # 2 Secondary Branching Streets
        for b in range(2):
            x_branch = 120 + b * 220 + int(np.random.uniform(-20, 20))
            branch_pts = [(x_branch, y_hwy_pixel), (x_branch, tile_size - 20)]
            cv2.line(
                img,
                branch_pts[0],
                branch_pts[1],
                color=(80, 80, 85),
                thickness=8,
            )

            bp1_geo = (origin_x + x_branch * pixel_size, origin_y - y_hwy_pixel * pixel_size)
            bp2_geo = (origin_x + x_branch * pixel_size, origin_y - (tile_size - 20) * pixel_size)
            lines_geo.append(LineString([bp1_geo, bp2_geo]))

        # 3. Simulate Tree Canopy Occlusion (Green patches covering road sections)
        for _ in range(4):
            tree_cx = int(np.random.uniform(100, tile_size - 100))
            tree_cy = y_hwy_pixel + int(np.random.uniform(-10, 10))
            radius = int(np.random.uniform(14, 24))
            cv2.circle(
                img,
                (tree_cx, tree_cy),
                radius,
                color=(40, 120, 45),  # Deep green foliage
                thickness=-1,
            )

        # Transpose back to (bands, height, width) for rasterio
        bands_data = np.transpose(img, (2, 0, 1))

        # Write GeoTIFF
        tif_path = os.path.join(output_dir, f"{tile_id}.tif")
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=tile_size,
            width=tile_size,
            count=3,
            dtype=rasterio.uint8,
            crs="EPSG:32611",
            transform=transform,
        ) as dst:
            dst.write(bands_data)

        # Write GeoJSON
        geojson_path = os.path.join(output_dir, f"{tile_id}.geojson")
        gdf = gpd.GeoDataFrame({"road_id": [f"road_{k}" for k in range(len(lines_geo))], "geometry": lines_geo}, crs="EPSG:32611")
        gdf.to_file(geojson_path, driver="GeoJSON")

    print(f"Successfully generated {num_tiles} sample pairs in {output_dir}!")


if __name__ == "__main__":
    generate_synthetic_tiles()
