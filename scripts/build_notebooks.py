import json
import os

def create_notebook(cells, output_path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {output_path}")

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

# -------------------------------------------------------------
# Notebook 1: Data Exploration & Simulation
# -------------------------------------------------------------
nb1_cells = [
    make_markdown_cell("""# TRACE: Data Exploration & Incomplete Basemap Simulation

This notebook visualizes:
1. High-resolution satellite GeoTIFF tiles and OpenStreetMap GeoJSON linestrings.
2. Basemap degradation simulation strategies (Random Edge Dropout, Suburban Sprawl, Disaster Severance).
3. The 4-channel multi-modal tensor stack `[Red, Green, Blue, Prior_Map]`.
"""),
    make_code_cell("""import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd

# Add project root to sys.path
sys.path.append("..")
from src.data.rasterizer import rasterize_linestrings
from src.data.degradation import simulate_incomplete_map
from src.common.geo_utils import geo_to_pixel"""),
    make_markdown_cell("## 1. Load Satellite Tile & Vector Road Linestrings"),
    make_code_cell("""img_path = "../data/raw/synthetic_tile_001.tif"
geo_path = "../data/raw/synthetic_tile_001.geojson"

with rasterio.open(img_path) as src:
    img_rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
    transform = src.transform
    crs = src.crs

gdf = gpd.read_file(geo_path)
print(f"Loaded imagery with shape: {img_rgb.shape}, CRS: {crs}")
print(f"Loaded {len(gdf)} vector road linestrings.")"""),
    make_markdown_cell("## 2. Simulate Incomplete Prior Map (30% Road Dropout)"),
    make_code_cell("""prior_gdf, residual_gdf = simulate_incomplete_map(gdf, strategy="random_drop", drop_ratio=0.30, seed=42)
print(f"Known Prior Roads (70%): {len(prior_gdf)} segments")
print(f"Missing Residual Roads (30%): {len(residual_gdf)} segments (What TRACE must discover!)")"""),
    make_markdown_cell("## 3. Visualize Ground Truth vs Incomplete Basemap"),
    make_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Subplot 1: Raw Satellite Image
axes[0].imshow(img_rgb)
axes[0].set_title("1. Raw Satellite Imagery (RGB)")
axes[0].axis("off")

# Subplot 2: Known Prior Map (Channel 4)
axes[1].imshow(img_rgb)
for geom in prior_gdf.geometry:
    coords = [geo_to_pixel(x, y, transform) for x, y in geom.coords]
    xs, ys = zip(*coords)
    axes[1].plot(xs, ys, color="cyan", linewidth=3, label="Prior Map")
axes[1].set_title("2. Existing Incomplete Map (70% Prior)")
axes[1].axis("off")

# Subplot 3: Missing Roads (Ground Truth Target)
axes[2].imshow(img_rgb)
for geom in residual_gdf.geometry:
    coords = [geo_to_pixel(x, y, transform) for x, y in geom.coords]
    xs, ys = zip(*coords)
    axes[2].plot(xs, ys, color="red", linewidth=3, linestyle="--", label="Target Residual")
axes[2].set_title("3. Target Residual Roads (30% to Discover)")
axes[2].axis("off")

plt.tight_layout()
plt.show()"""),
    make_markdown_cell("## 4. Inspect the 4-Channel Input Tensor Stack"),
    make_code_cell("""prior_mask = rasterize_linestrings(prior_gdf, shape=(512, 512), transform=transform, line_thickness_pixels=4, gaussian_sigma=1.5)
target_mask = rasterize_linestrings(residual_gdf, shape=(512, 512), transform=transform, line_thickness_pixels=4, gaussian_sigma=0.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(prior_mask, cmap="magma")
axes[0].set_title("Channel 4: Prior Map Mask (with Gaussian Tolerance)")
axes[0].axis("off")

axes[1].imshow(target_mask, cmap="inferno")
axes[1].set_title("Target Label: Residual Centerline Mask")
axes[1].axis("off")

plt.tight_layout()
plt.show()""")
]

# -------------------------------------------------------------
# Notebook 2: Cloud Training on Google Colab / Kaggle
# -------------------------------------------------------------
nb2_cells = [
    make_markdown_cell("""# TRACE: Google Colab & Cloud GPU Training Pipeline

Run this notebook in Google Colab with an **NVIDIA T4 GPU** to train the complete 4-channel D-LinkNet model with **clDice** connectivity loss.
"""),
    make_markdown_cell("## Step 1: Check GPU Runtime"),
    make_code_cell("""import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Running on CPU. In menu: Runtime -> Change runtime type -> Select T4 GPU.")"""),
    make_markdown_cell("## Step 2: Mount Google Drive (To Save Checkpoints Permanently)"),
    make_code_cell("""from google.colab import drive
import os

drive.mount('/content/drive')
drive_checkpoint_dir = "/content/drive/MyDrive/TRACE_Checkpoints"
os.makedirs(drive_checkpoint_dir, exist_ok=True)
print(f"Checkpoints will sync to: {drive_checkpoint_dir}")"""),
    make_markdown_cell("## Step 3: Clone Repository & Install Dependencies"),
    make_code_cell("""!git clone https://github.com/<your-username>/TRACE.git /content/TRACE
%cd /content/TRACE
!pip install -q -r requirements.txt
!pip install -q pytest"""),
    make_markdown_cell("## Step 4: Run Component Unit Tests"),
    make_code_cell("""!pytest tests/ -v"""),
    make_markdown_cell("## Step 5: Generate Sample Data (or Symlink SpaceNet)"),
    make_code_cell("""!python scripts/generate_synthetic_sample.py"""),
    make_markdown_cell("## Step 6: Launch Model Training"),
    make_code_cell("""!python train.py --epochs 60 --batch_size 16 --lr 0.0002"""),
    make_markdown_cell("## Step 7: Run Evaluation & Copy Best Weights to Google Drive"),
    make_code_cell("""!python evaluate.py --checkpoint checkpoints/best_model.pth
!cp checkpoints/best_model.pth /content/drive/MyDrive/TRACE_Checkpoints/best_model_run1.pth
print("Saved best checkpoint permanently to Google Drive!")""")
]

# -------------------------------------------------------------
# Notebook 3: Inference, Topological Snapping & APLS Routing
# -------------------------------------------------------------
nb3_cells = [
    make_markdown_cell("""# TRACE: End-to-End Inference, Vector Snapping & APLS Routing

This notebook demonstrates:
1. Running inference with the trained model checkpoint.
2. Converting probability heatmaps into vector centerlines.
3. Snapping newly discovered roads into the master OpenStreetMap graph.
4. Running Dijkstra vehicle shortest-path routing over the newly completed road network!
"""),
    make_code_cell("""import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd

sys.path.append("..")
from src.pipeline.inference import TRACEInferencePipeline
from src.evaluation.apls import compute_apls"""),
    make_markdown_cell("## 1. Run TRACE Inference Pipeline"),
    make_code_cell("""pipeline = TRACEInferencePipeline(
    checkpoint_path="../checkpoints/best_model.pth",
    prob_threshold=0.50,
    snap_search_radius=30.0,
    spur_length_meters=15.0
)

image_path = "../data/raw/synthetic_tile_001.tif"
prior_path = "../data/raw/synthetic_tile_001.geojson"
output_geojson_path = "../data/outputs/updated_network.geojson"

unified_graph, out_gdf = pipeline.run(
    image_path=image_path,
    prior_geojson_path=prior_path,
    output_geojson_path=output_geojson_path
)

print(f"Successfully generated unified road network!")
print(f"Total Nodes: {len(unified_graph.nodes)}")
print(f"Total Edges: {len(unified_graph.edges)}")"""),
    make_markdown_cell("## 2. Visualize Graph Overlay on Satellite Image"),
    make_code_cell("""import rasterio

with rasterio.open(image_path) as src:
    img = src.read([1, 2, 3]).transpose(1, 2, 0)
    transform = src.transform

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)

for u, v in unified_graph.edges():
    p1 = unified_graph.nodes[u]["pos_geo"]
    p2 = unified_graph.nodes[v]["pos_geo"]
    # Convert geo coordinates to pixel space for plotting
    inv_t = ~transform
    c1, r1 = inv_t * p1
    c2, r2 = inv_t * p2
    ax.plot([c1, c2], [r1, r2], color="yellow", linewidth=2.5)

ax.set_title("Unified Traversable Road Graph Overlaid on Satellite Imagery", fontsize=14)
ax.axis("off")
plt.show()"""),
    make_markdown_cell("## 3. Dijkstra Shortest-Path Vehicle Routing Test"),
    make_code_cell("""nodes = list(unified_graph.nodes())
start_node = nodes[0]
end_node = nodes[-1]

if nx.has_path(unified_graph, start_node, end_node):
    path = nx.shortest_path(unified_graph, start_node, end_node, weight="weight")
    path_len = nx.shortest_path_length(unified_graph, start_node, end_node, weight="weight")
    print(f"Routing Successful!")
    print(f"Traversed {len(path)} junction nodes across {path_len:.2f} meters.")
else:
    print("Path not found - checking graph connectivity.")""")
]

os.makedirs("notebooks", exist_ok=True)
create_notebook(nb1_cells, "notebooks/01_data_exploration_and_simulation.ipynb")
create_notebook(nb2_cells, "notebooks/02_train_colab.ipynb")
create_notebook(nb3_cells, "notebooks/03_inference_and_apls_benchmark.ipynb")
