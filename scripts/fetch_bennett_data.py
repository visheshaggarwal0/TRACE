import os
import math
import io
import ssl
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def fetch_satellite_image(min_lat, min_lon, max_lat, max_lon, zoom=17):
    x_min, y_min = deg2num(max_lat, min_lon, zoom)
    x_max, y_max = deg2num(min_lat, max_lon, zoom)

    x_tiles = list(range(x_min, x_max + 1))
    y_tiles = list(range(y_min, y_max + 1))

    tile_w, tile_h = 256, 256
    total_w = len(x_tiles) * tile_w
    total_h = len(y_tiles) * tile_h

    stitched = Image.new("RGB", (total_w, total_h))
    print(f"Downloading {len(x_tiles) * len(y_tiles)} satellite tiles at Zoom level {zoom}...")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TRACE/1.0"}
    ctx = ssl._create_unverified_context()

    for ix, x in enumerate(x_tiles):
        for iy, y in enumerate(y_tiles):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                    tile_img = Image.open(io.BytesIO(resp.read()))
                    stitched.paste(tile_img, (ix * tile_w, iy * tile_h))
            except Exception as e:
                print(f"Warning: Failed to fetch tile ({zoom}, {x}, {y}): {e}")

    nw_lat, nw_lon = num2deg(x_min, y_min, zoom)
    se_lat, se_lon = num2deg(x_max + 1, y_max + 1, zoom)

    return stitched, (se_lat, nw_lon, nw_lat, se_lon)


def fetch_osm_roads_direct_api(min_lat, min_lon, max_lat, max_lon):
    """Fetches real roads directly from official OpenStreetMap REST API."""
    print("Querying official OpenStreetMap API for road vectors...")
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={min_lon},{min_lat},{max_lon},{max_lat}"
    headers = {"User-Agent": "TRACE_Research/1.0"}
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=25, context=ctx) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)

    # 1. Parse nodes
    nodes = {}
    for node in root.findall(".//node"):
        nid = node.get("id")
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        nodes[nid] = (lon, lat)

    # 2. Parse ways tagged with highway
    lines = []
    road_types = []
    names = []

    for way in root.findall(".//way"):
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        if "highway" in tags:
            nd_refs = [nd.get("ref") for nd in way.findall("nd")]
            coords = [nodes[ref] for ref in nd_refs if ref in nodes]
            if len(coords) >= 2:
                lines.append(LineString(coords))
                road_types.append(tags.get("highway", "road"))
                names.append(tags.get("name", "Unnamed"))

    gdf = gpd.GeoDataFrame({"name": names, "highway": road_types, "geometry": lines}, crs="EPSG:4326")
    print(f"Retrieved {len(gdf)} real OpenStreetMap road segments around Bennett University!")
    return gdf


def main():
    min_lat, min_lon = 28.444, 77.575
    max_lat, max_lon = 28.457, 77.592

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/outputs", exist_ok=True)

    tif_path = os.path.join(output_dir, "bennett_university_sat.tif")

    if os.path.exists(tif_path):
        print(f"Found existing satellite GeoTIFF at {tif_path}, loading...")
        with rasterio.open(tif_path) as src:
            img_arr = src.read([1, 2, 3]).transpose(1, 2, 0)
            img = Image.fromarray(img_arr)
            actual_min_lon, actual_min_lat, actual_max_lon, actual_max_lat = src.bounds
            transform = src.transform
    else:
        img, (actual_min_lat, actual_min_lon, actual_max_lat, actual_max_lon) = fetch_satellite_image(
            min_lat, min_lon, max_lat, max_lon, zoom=17
        )
        img_np = np.array(img).transpose(2, 0, 1)
        transform = from_bounds(actual_min_lon, actual_min_lat, actual_max_lon, actual_max_lat, img.width, img.height)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=img.height,
            width=img.width,
            count=3,
            dtype=rasterio.uint8,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(img_np)
        print(f"Saved satellite GeoTIFF to: {tif_path}")

    # 2. Fetch OpenStreetMap Roads via Direct REST API
    osm_gdf = fetch_osm_roads_direct_api(actual_min_lat, actual_min_lon, actual_max_lat, actual_max_lon)
    geojson_path = os.path.join(output_dir, "bennett_university_roads.geojson")
    osm_gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"Saved OSM roads GeoJSON to: {geojson_path}")

    # 3. Create High-Res Visualization
    print("Generating demo visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Subplot 1: Pure Satellite Image
    axes[0].imshow(img)
    axes[0].set_title("1. Real Satellite Imagery (Greater Noida / Bennett University)", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Subplot 2: Satellite Image + OpenStreetMap Road Overlay
    axes[1].imshow(img)
    inv_transform = ~transform
    for _, row in osm_gdf.iterrows():
        coords = list(row.geometry.coords)
        pixel_coords = [inv_transform @ (lon, lat) for lon, lat in coords]
        xs, ys = zip(*pixel_coords)
        axes[1].plot(xs, ys, color="#00FFCC", linewidth=2.5, alpha=0.9)

    axes[1].set_title(f"2. Satellite Photo + Live OpenStreetMap Roads ({len(osm_gdf)} segments)", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    vis_path = "data/outputs/bennett_analysis.png"
    plt.tight_layout()
    plt.savefig(vis_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved analysis plot to: {vis_path}")


if __name__ == "__main__":
    main()
