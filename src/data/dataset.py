import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import geopandas as gpd
from affine import Affine

from .rasterizer import rasterize_linestrings
from .degradation import simulate_incomplete_map


class TRACEDataset(Dataset):
    """PyTorch Dataset for Map-Conditioned Residual Road Extraction.

    Each sample consists of:
        - Input: 4-channel tensor [Red, Green, Blue, Prior_Map_Mask] of shape (4, H, W).
        - Target: 1-channel tensor [Residual_Road_Mask] of shape (1, H, W).
    """

    def __init__(
        self,
        image_paths: List[str],
        geojson_paths: List[str],
        degradation_strategy: str = "random_drop",
        drop_ratio: float = 0.30,
        buffer_width_pixels: int = 4,
        gaussian_sigma: float = 1.5,
        transform=None,
    ):
        assert len(image_paths) == len(geojson_paths), "Mismatch between images and GeoJSON paths."
        self.image_paths = image_paths
        self.geojson_paths = geojson_paths
        self.degradation_strategy = degradation_strategy
        self.drop_ratio = drop_ratio
        self.buffer_width_pixels = buffer_width_pixels
        self.gaussian_sigma = gaussian_sigma
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        img_path = self.image_paths[idx]
        geo_path = self.geojson_paths[idx]

        with rasterio.open(img_path) as src:
            # Read first 3 channels (RGB) normalized to [0, 1]
            img_data = src.read([1, 2, 3]).astype(np.float32) / 255.0
            height, width = src.height, src.width
            affine_transform = src.transform
            crs = src.crs

        # Read Ground Truth Vector Roads
        if os.path.exists(geo_path) and os.path.getsize(geo_path) > 0:
            full_gdf = gpd.read_file(geo_path)
            if full_gdf.crs != crs and crs is not None:
                full_gdf = full_gdf.to_crs(crs)
        else:
            full_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=crs)

        # Simulate Incomplete Prior Map
        prior_gdf, residual_gdf = simulate_incomplete_map(
            full_gdf, strategy=self.degradation_strategy, drop_ratio=self.drop_ratio
        )

        # Rasterize Prior Channel (Channel 4) with Gaussian smoothing
        prior_mask = rasterize_linestrings(
            prior_gdf,
            shape=(height, width),
            transform=affine_transform,
            line_thickness_pixels=self.buffer_width_pixels,
            gaussian_sigma=self.gaussian_sigma,
        )

        # Rasterize Target Residual Mask (What the network must learn to discover)
        target_mask = rasterize_linestrings(
            residual_gdf,
            shape=(height, width),
            transform=affine_transform,
            line_thickness_pixels=self.buffer_width_pixels,
            gaussian_sigma=0.0,
        )

        # Stack into 4-channel input: [R, G, B, Prior]
        prior_expanded = np.expand_dims(prior_mask, axis=0)
        input_array = np.concatenate([img_data, prior_expanded], axis=0)  # Shape (4, H, W)
        target_array = np.expand_dims(target_mask, axis=0)               # Shape (1, H, W)

        # Basic spatial augmentations if provided
        if self.transform is not None:
            # albumentations expects HWC
            transposed_in = np.transpose(input_array, (1, 2, 0))
            transposed_tgt = np.transpose(target_array, (1, 2, 0))
            augmented = self.transform(image=transposed_in, mask=transposed_tgt)
            input_array = np.transpose(augmented["image"], (2, 0, 1))
            target_array = np.transpose(augmented["mask"], (2, 0, 1))

        input_tensor = torch.from_numpy(input_array.copy()).float()
        target_tensor = torch.from_numpy(target_array.copy()).float()

        metadata = {
            "image_path": img_path,
            "affine_transform": list(affine_transform)[:6],
            "crs": str(crs),
            "original_shape": (height, width),
        }

        return input_tensor, target_tensor, metadata
