"""Adaptive Prior Gated D-LinkNet (APG-DLinkNet) V2: Zero-Init FiLM Modulation.

Key Improvements over V1:
1. Zero-Convolution Initialization (ControlNet paradigm):
   Initializes modulation parameters (gamma, beta) to exactly zero.
   Guarantees that at Step 0, F_fused = (1 + 0) * F_sat + 0 == F_sat.
   Eliminates random noise poisoning of ImageNet pretrained features.
2. Replicate Padding:
   Replaces zero-padding with replicate padding to eliminate border false-positive artifacts.
3. Feature-wise Linear Modulation (FiLM):
   Prior map adaptively scales and shifts satellite features instead of naive addition.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

from .dlinknet import DilatedConvBlock, DecoderBlock


class PriorEncoder(nn.Module):
    """Lightweight convolutional encoder for 1-channel rasterized prior maps."""

    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, padding_mode="replicate", bias=False),  # H/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1, padding_mode="replicate", bias=False),  # H/4
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ZeroInitFiLMGate(nn.Module):
    """Zero-Initialized Feature-wise Linear Modulation (FiLM) with Replicate Padding."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, padding_mode="replicate", bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.zero_conv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)

        # STRICT ZERO INITIALIZATION
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, feat_sat: torch.Tensor, feat_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([feat_sat, feat_prior], dim=1)
        hidden = self.conv_in(concat)
        mod = self.zero_conv(hidden)  # [B, 2*C, H, W]
        gamma, beta = torch.chunk(mod, 2, dim=1)

        # Stable FiLM formulation: scale + shift
        fused = (1.0 + torch.tanh(gamma)) * feat_sat + beta

        # Gate map for visual telemetry
        gate_map = torch.sigmoid(torch.norm(mod, dim=1, keepdim=True))
        return fused, gate_map


class APGDLinkNet34(nn.Module):
    """Adaptive Prior Gated D-LinkNet34 V2 with Zero-Init FiLM Modulation."""

    def __init__(self, in_channels: int = 4, out_channels: int = 1, pretrained: bool = False):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        base = resnet34(weights=weights)

        # Satellite RGB Stream (channels 0:3)
        self.sat_conv1 = base.conv1
        self.sat_bn1 = base.bn1
        self.sat_relu = base.relu
        self.sat_maxpool = base.maxpool

        # Dedicated Prior Map Stream (channel 3)
        self.prior_encoder = PriorEncoder(out_channels=64)

        # Zero-Initialized FiLM Gating
        self.apg = ZeroInitFiLMGate(channels=64)

        # Shared Deep Stages (ResNet34)
        self.layer1 = base.layer1  # 64ch -> 64ch
        self.layer2 = base.layer2  # 64ch -> 128ch
        self.layer3 = base.layer3  # 128ch -> 256ch
        self.layer4 = base.layer4  # 256ch -> 512ch

        # Central Dilated Bottleneck (D-Link)
        self.dlink = DilatedConvBlock(512, 512)

        # Decoders with Skip-Connections
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.decoder0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.last_gate_map: torch.Tensor | None = None

    def get_parameter_groups(self, lr_backbone: float = 5e-5, lr_new: float = 5e-4):
        """Differential learning rate parameter grouping."""
        backbone_names = {"sat_conv1", "sat_bn1", "layer1", "layer2", "layer3", "layer4"}
        backbone_params = []
        new_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(name.startswith(b) for b in backbone_names):
                backbone_params.append(param)
            else:
                new_params.append(param)

        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
            {"params": new_params, "lr": lr_new, "weight_decay": 1e-4},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split 4-channel input into RGB satellite and 1-channel prior map
        x_sat = x[:, :3, :, :]
        x_prior = x[:, 3:4, :, :]

        # 1. Satellite Stream
        e0 = self.sat_relu(self.sat_bn1(self.sat_conv1(x_sat)))
        p_sat = self.sat_maxpool(e0)  # H/4, W/4, 64 channels

        # 2. Prior Map Stream
        p_prior = self.prior_encoder(x_prior)  # H/4, W/4, 64 channels

        # 3. Dynamic Zero-Init FiLM Gating Fusion
        p_fused, gate = self.apg(p_sat, p_prior)
        self.last_gate_map = gate

        # 4. Deep Feature Extraction
        e1 = self.layer1(p_fused)  # H/4, 64ch
        e2 = self.layer2(e1)       # H/8, 128ch
        e3 = self.layer3(e2)       # H/16, 256ch
        e4 = self.layer4(e3)       # H/32, 512ch

        # 5. Dilated Center Block
        center = self.dlink(e4)

        # 6. Decoder with Skip Connections
        d4 = self.decoder4(center, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)
        d0 = self.decoder0(d1)

        out = self.final_conv(d0)
        return out
