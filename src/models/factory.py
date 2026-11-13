"""Model factory for TRACE perception architectures.

Supports dynamic instantiation of supported architectures:
- 'dlinknet34' / 'dlinknet': D-LinkNet34 with ResNet34 backbone & dilated bottleneck
- 'unet': Baseline 4-channel U-Net with full skip connections
- 'deeplabv3plus' / 'deeplab': DeepLabv3+ with ResNet34 backbone & ASPP pyramid
"""
import torch.nn as nn
from typing import Dict, Any

from .dlinknet import DLinkNet34
from .unet import UNet4Channel
from .deeplabv3plus import DeepLabV3Plus
from .apg_dlinknet import APGDLinkNet34


SUPPORTED_MODELS = {
    "dlinknet34": DLinkNet34,
    "dlinknet": DLinkNet34,
    "unet": UNet4Channel,
    "deeplabv3plus": DeepLabV3Plus,
    "deeplab": DeepLabV3Plus,
    "apg_dlinknet": APGDLinkNet34,
    "apg_dlinknet34": APGDLinkNet34,
    "apg": APGDLinkNet34,
}


def build_model(
    model_name: str = "dlinknet34",
    in_channels: int = 4,
    out_channels: int = 1,
    pretrained: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Build and return a road perception segmentation model.

    Args:
        model_name: Name of model architecture ('dlinknet34', 'unet', 'deeplabv3plus').
        in_channels: Number of input channels (4 for RGB + Prior Map).
        out_channels: Number of output channels (1 for binary road logits).
        pretrained: Whether to load ImageNet pretrained weights (where applicable).
        **kwargs: Additional model-specific hyperparameters.

    Returns:
        nn.Module: Initialized PyTorch model.
    """
    key = model_name.lower().strip()
    if key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model architecture: '{model_name}'. "
            f"Supported architectures are: {list(SUPPORTED_MODELS.keys())}"
        )

    model_cls = SUPPORTED_MODELS[key]

    if key in ("unet",):
        # UNet doesn't use torchvision pretrained weights
        base_features = kwargs.get("base_features", 32)
        return model_cls(in_channels=in_channels, out_channels=out_channels, base_features=base_features)

    return model_cls(in_channels=in_channels, out_channels=out_channels, pretrained=pretrained)
