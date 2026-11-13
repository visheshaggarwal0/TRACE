import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feat = self.conv(self.gap(x))
        if self.training and feat.shape[0] == 1:
            feat = self.relu(feat)
        else:
            feat = self.relu(self.bn(feat))
        return F.interpolate(feat, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module from DeepLabv3+."""

    def __init__(self, in_channels: int, out_channels: int = 256, atrous_rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ architecture adapted for 4-channel Map-Conditioned Road Extraction.

    Uses ResNet34 as backbone with:
      - 4-channel conv1 input surgery
      - ASPP module on deep bottleneck features
      - Low-level skip connection fusion in decoder
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1, pretrained: bool = False):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        base = resnet34(weights=weights)

        # 4-channel input surgery
        orig_conv1 = base.conv1
        self.conv1 = nn.Conv2d(
            in_channels,
            orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=orig_conv1.stride,
            padding=orig_conv1.padding,
            bias=False,
        )

        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = orig_conv1.weight
            if in_channels > 3:
                self.conv1.weight[:, 3:, :, :] = torch.mean(orig_conv1.weight, dim=1, keepdim=True)

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Encoder stages
        self.layer1 = base.layer1  # 64 channels, stride 4 relative to input
        self.layer2 = base.layer2  # 128 channels, stride 8
        self.layer3 = base.layer3  # 256 channels, stride 16
        self.layer4 = base.layer4  # 512 channels, stride 32

        # ASPP module at the deepest features
        self.aspp = ASPP(in_channels=512, out_channels=256, atrous_rates=(6, 12, 18))

        # Low-level feature projection (from layer1: 64 channels -> 48 channels)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Decoder classifier head
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # Encoder
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        low_level = self.layer1(x)  # H/4, W/4
        x = self.layer2(low_level)  # H/8, W/8
        x = self.layer3(x)          # H/16, W/16
        high_level = self.layer4(x) # H/32, W/32

        # ASPP
        aspp_out = self.aspp(high_level)

        # Upsample ASPP features to match low_level size (H/4, W/4)
        aspp_up = F.interpolate(
            aspp_out, size=low_level.shape[-2:], mode="bilinear", align_corners=False
        )

        # Project low_level features
        low_proj = self.low_level_proj(low_level)

        # Concatenate & decode
        fused = torch.cat([aspp_up, low_proj], dim=1)
        decoded = self.decoder(fused)

        # Final upsample to original resolution (H, W)
        output = F.interpolate(decoded, size=input_size, mode="bilinear", align_corners=False)
        return output
