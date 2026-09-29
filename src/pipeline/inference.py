import os
from typing import Optional, Tuple
import numpy as np
import torch
import rasterio
import geopandas as gpd
import networkx as nx
from affine import Affine

from ..data.rasterizer import rasterize_linestrings
from ..vectorization.skeletonizer import extract_skeleton_graph
from ..vectorization.graph_contraction import (
    contract_graph,
    prune_short_spurs,
    collapse_junction_clusters,
)
from ..vectorization.snapping_engine import snap_subgraph_to_master, export_graph_to_geojson
from ..models.dlinknet import DLinkNet34


class TRACEInferencePipeline:
    """Production inference pipeline for updating road networks from satellite imagery."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        prob_threshold: float = 0.50,
        spur_length_meters: float = 15.0,
        junction_merge_radius: float = 10.0,
        snap_search_radius: float = 30.0,
    ):
        self.device = torch.device(device)
        self.prob_threshold = prob_threshold
        self.spur_length_meters = spur_length_meters
        self.junction_merge_radius = junction_merge_radius
        self.snap_search_radius = snap_search_radius

        self.model = DLinkNet34(in_channels=4, out_channels=1, pretrained=False)
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def run(
        self,
        image_path: str,
        prior_geojson_path: str,
        output_geojson_path: str,
    ) -> Tuple[nx.Graph, gpd.GeoDataFrame]:
        """Runs the full TRACE pipeline:

        Input: GeoTIFF satellite image + prior GeoJSON vector map.
        Output: Merged, fully connected GeoJSON road network.
        """
        # 1. Read satellite imagery
        with rasterio.open(image_path) as src:
            img_rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0
            height, width = src.height, src.width
            affine_transform = src.transform
            crs = src.crs

        # 2. Read prior GeoJSON and build master graph
        if os.path.exists(prior_geojson_path) and os.path.getsize(prior_geojson_path) > 0:
            prior_gdf = gpd.read_file(prior_geojson_path)
            if prior_gdf.crs != crs and crs is not None:
                prior_gdf = prior_gdf.to_crs(crs)
        else:
            prior_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=crs)

        # 3. Rasterize prior channel
        prior_mask = rasterize_linestrings(
            prior_gdf,
            shape=(height, width),
            transform=affine_transform,
            line_thickness_pixels=4,
            gaussian_sigma=1.5,
        )

        # 4. Prepare 4-channel tensor and run deep model
        prior_expanded = np.expand_dims(prior_mask, axis=0)
        input_tensor = (
            torch.from_numpy(np.concatenate([img_rgb, prior_expanded], axis=0))
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.model(input_tensor)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        # 5. Topological Vectorization
        raw_graph, _ = extract_skeleton_graph(
            prob_map, affine_transform, prob_threshold=self.prob_threshold
        )
        contracted_graph = contract_graph(raw_graph)
        pruned_graph = prune_short_spurs(contracted_graph, min_length_meters=self.spur_length_meters)
        new_road_graph = collapse_junction_clusters(
            pruned_graph, merge_radius_meters=self.junction_merge_radius
        )

        # 6. Build master graph from prior GeoDataFrame
        master_graph = nx.Graph()
        for idx, row in prior_gdf.iterrows():
            geom = row.geometry
            if geom is not None and geom.geom_type == "LineString":
                coords = list(geom.coords)
                for i in range(len(coords) - 1):
                    p1, p2 = coords[i], coords[i + 1]
                    dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    u_id = f"osm_{p1[0]:.4f}_{p1[1]:.4f}"
                    v_id = f"osm_{p2[0]:.4f}_{p2[1]:.4f}"
                    master_graph.add_node(u_id, pos_geo=p1)
                    master_graph.add_node(v_id, pos_geo=p2)
                    master_graph.add_edge(u_id, v_id, weight=dist)

        # 7. Snap newly discovered road subgraph to master graph
        final_unified_graph = snap_subgraph_to_master(
            new_graph=new_road_graph,
            master_graph=master_graph,
            search_radius_meters=self.snap_search_radius,
        )

        # 8. Export to GeoJSON
        crs_str = str(crs) if crs else "EPSG:32611"
        out_gdf = export_graph_to_geojson(
            final_unified_graph, output_geojson_path=output_geojson_path, crs=crs_str
        )

        return final_unified_graph, out_gdf
