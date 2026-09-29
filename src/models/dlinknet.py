import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class DilatedConvBlock(nn.Module):
    """Central dilated convolution module (D-Link) to capture wide receptive fields without pooling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.d1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.d2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.d4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=4, padding=4)
        self.d8 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=8, padding=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        d1 = self.relu(self.d1(x))
        d2 = self.relu(self.d2(x))
        d4 = self.relu(self.d4(x))
        d8 = self.relu(self.d8(x))
        return x + d1 + d2 + d4 + d8


class DecoderBlock(nn.Module):
    """Upsampling decoder block with skip-connection fusion."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DLinkNet34(nn.Module):
    """D-LinkNet34 adapted for 4-channel Map-Conditioned Road Extraction.

    Input Channels: [Red, Green, Blue, Prior_Map]
    Output Channels: 1 (Probability heatmap of missing road centerlines)
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        base = resnet34(weights=weights)

        # Modify first layer for 4 channels
        original_conv1 = base.conv1
        self.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )

        with torch.no_grad():
            # Copy pretrained RGB weights to first 3 channels
            self.conv1.weight[:, :3, :, :] = original_conv1.weight
            if in_channels > 3:
                # Initialize 4th channel (prior map) as the mean of RGB channels
                self.conv1.weight[:, 3:, :, :] = torch.mean(original_conv1.weight, dim=1, keepdim=True)

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Encoder stages
        self.encoder1 = base.layer1  # 64 channels
        self.encoder2 = base.layer2  # 128 channels
        self.encoder3 = base.layer3  # 256 channels
        self.encoder4 = base.layer4  # 512 channels

        # Center dilated block
        self.center = DilatedConvBlock(512, 512)

        # Decoder stages
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final head
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial stem
        x0 = self.relu(self.bn1(self.conv1(x)))
        x_pool = self.maxpool(x0)

        # Encoder
        e1 = self.encoder1(x_pool)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Central dilation
        c = self.center(e4)

        # Decoder with skip connections
        d4 = self.decoder4(c, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)

        out = self.final(d1)
        return out
