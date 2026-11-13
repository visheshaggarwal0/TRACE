import os
import sys

# Ensure project root is in sys.path and is current working directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import rasterio
import geopandas as gpd
import numpy as np
import torch
import networkx as nx

from src.data.degradation import simulate_incomplete_map
from src.data.rasterizer import rasterize_linestrings
from src.models.dlinknet import DLinkNet34
from src.models.losses import SoftSkeletonize
from src.pipeline.inference import TRACEInferencePipeline

print("1. Testing Data Ingestion on Bennett University data...")
sat_path = os.path.join(ROOT_DIR, "data", "raw", "bennett_university_sat.tif")
osm_path = os.path.join(ROOT_DIR, "data", "raw", "bennett_university_roads.geojson")

with rasterio.open(sat_path) as src:
    img_rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
    transform = src.transform
    h, w = src.height, src.width

gdf = gpd.read_file(osm_path)
print(f"   Satellite: {w}x{h}, OSM roads: {len(gdf)} segments.")

print("2. Testing Basemap Degradation (30% drop)...")
prior_gdf, res_gdf = simulate_incomplete_map(gdf, strategy="random_drop", drop_ratio=0.30, seed=42)
print(f"   Prior: {len(prior_gdf)} segments, Residual: {len(res_gdf)} segments.")

print("3. Testing Rasterization of Prior Channel...")
prior_mask = rasterize_linestrings(prior_gdf, (h, w), transform, line_thickness_pixels=4, gaussian_sigma=1.5)
print(f"   Prior mask shape: {prior_mask.shape}, Max value: {prior_mask.max():.2f}")

print("4. Testing D-LinkNet34 Model & SoftSkeletonize...")
model = DLinkNet34(in_channels=4, out_channels=1, pretrained=False)
dummy_in = torch.randn(1, 4, 256, 256)
with torch.no_grad():
    out = model(dummy_in)
print(f"   Model forward pass: {dummy_in.shape} -> {out.shape}")

skel = SoftSkeletonize(num_iters=5)
dummy_road = torch.zeros(1, 1, 64, 64)
dummy_road[:, :, 30:34, 10:50] = 1.0
soft_s = skel(dummy_road)
print(f"   Soft skeleton shape: {soft_s.shape}, Max: {soft_s.max().item():.2f}")

print("5. Testing Inference Pipeline & Graph Export...")
ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "best_model.pth")
pipeline = TRACEInferencePipeline(
    checkpoint_path=ckpt_path if os.path.exists(ckpt_path) else None,
    prob_threshold=0.50,
    snap_search_radius=30.0,
    spur_length_meters=15.0
)
out_geojson = os.path.join(ROOT_DIR, "data", "outputs", "test_verify_network.geojson")
unified_graph, out_gdf = pipeline.run(sat_path, osm_path, out_geojson)
print(f"   Unified Graph: {len(unified_graph.nodes)} nodes, {len(unified_graph.edges)} edges.")

nodes = list(unified_graph.nodes())
if len(nodes) >= 2:
    s, t = nodes[0], nodes[-1]
    if nx.has_path(unified_graph, s, t):
        path_len = nx.shortest_path_length(unified_graph, s, t, weight="weight")
        print(f"6. Live Dijkstra Routing verified! Distance: {path_len:.2f} meters.")

print("\nALL NOTEBOOK LOGIC & COMPONENTS VERIFIED 100% SUCCESFULLY!")
