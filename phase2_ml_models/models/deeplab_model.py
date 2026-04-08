"""
Glacier Melting Detection — DeepLabv3+ Model
=============================================
DeepLabv3+ for advanced glacier segmentation with atrous separable convolutions.

Architecture:
  Backbone: Modified ResNet-50 (first conv adapted for N input channels)
    - output_stride=16: replace stride-2 with dilation in layer3/layer4
  ASPP (Atrous Spatial Pyramid Pooling):
    - 1×1 conv
    - 3×3 dilated convs: rates [6, 12, 18]
    - global average pooling branch
    → fused to 256-channel feature
  Decoder:
    - Low-level features from layer1 (256→48 channels via 1×1)
    - Upsample ASPP ×4 → concat with low-level → 3×3 conv → 1×1 output
    - Final bilinear ×4 upsample to full resolution

Reference: Chen et al. 2018 — Encoder-Decoder with Atrous Separable Convolution
           for Semantic Image Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet34, ResNet50_Weights


# ══════════════════════════════════════════════════════════════════════════════
# ATROUS SEPARABLE CONVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

class AtrousSeparableConv(nn.Module):
    """
    Depthwise separable convolution with dilation.
    Reduces parameters vs standard dilated conv while keeping receptive field.
    """

    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, padding=None):
        super().__init__()
        if padding is None:
            padding = dilation
        # Depthwise
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel,
            padding=padding, dilation=dilation,
            groups=in_ch, bias=False,
        )
        # Pointwise
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn        = nn.BatchNorm2d(out_ch)
        self.relu      = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


# ══════════════════════════════════════════════════════════════════════════════
# ASPP MODULE
# ══════════════════════════════════════════════════════════════════════════════

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (DeepLab v3+).
    Captures multi-scale context without losing resolution.

    Branches:
      1. 1×1 conv (rate=1)
      2–4. Atrous separable convs at rates [6, 12, 18]
      5. Global average pooling

    All branches produce 256-channel outputs, then concatenated and projected.
    """

    def __init__(self, in_channels=2048, out_channels=256,
                 dilations=(6, 12, 18), dropout=0.1):
        super().__init__()

        # Branch 1: 1×1 conv
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branches 2–4: dilated convolutions
        self.atrous_convs = nn.ModuleList([
            nn.Sequential(
                AtrousSeparableConv(in_channels, out_channels, dilation=d),
            )
            for d in dilations
        ])

        # Branch 5: global average pooling
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Projection: concat 5 branches (5×256) → 256
        n_branches = 1 + len(dilations) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * n_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        branch1 = self.conv1x1(x)
        branches = [branch1] + [ac(x) for ac in self.atrous_convs]

        # Global pooling branch (upsample back to spatial size)
        gap_out = self.gap(x)
        gap_out = F.interpolate(gap_out, size=(h, w), mode="bilinear", align_corners=False)
        branches.append(gap_out)

        return self.project(torch.cat(branches, dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# DEEPLABV3+
# ══════════════════════════════════════════════════════════════════════════════

class DeepLabV3Plus(nn.Module):
    """
    DeepLabv3+ for multi-spectral glacier segmentation.

    Encoder: ResNet-50 backbone with output_stride=16.
             First conv is replaced to accept `in_channels` satellite bands.
    ASPP:    Multi-scale feature extraction.
    Decoder: Low-level features + ASPP fusion.
    Output:  (B, num_classes, H, W) — same resolution as input.
    """

    def __init__(self, in_channels=11, num_classes=4,
                 output_stride=16, aspp_dilations=(6, 12, 18),
                 dropout=0.1, pretrained_backbone=False):
        super().__init__()
        self.in_channels  = in_channels
        self.num_classes  = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        if pretrained_backbone:
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            backbone = resnet50(weights=None)

        # Replace first conv to accept satellite bands (was 3-channel RGB)
        backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained_backbone and in_channels == 3:
            # Keep pretrained weights only when in_channels matches
            pass
        else:
            nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Apply output_stride=16: modify layer3 and layer4 to use dilation
        # instead of stride-2 (keeps spatial resolution at 1/16 instead of 1/32)
        if output_stride == 16:
            layer3_stride = 1
            layer4_stride = 1
            layer3_dilation = 2
            layer4_dilation = 4
        else:  # output_stride == 8
            layer3_stride = 1
            layer4_stride = 1
            layer3_dilation = 2
            layer4_dilation = 2

        self._make_dilated(backbone.layer3, layer3_stride, layer3_dilation)
        self._make_dilated(backbone.layer4, layer4_stride, layer4_dilation)

        # Extract backbone stages
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 1/4, 256ch — used for low-level features
        self.layer2 = backbone.layer2   # 1/8, 512ch
        self.layer3 = backbone.layer3   # 1/16, 1024ch (dilated)
        self.layer4 = backbone.layer4   # 1/16, 2048ch (dilated)

        # ── ASPP ──────────────────────────────────────────────────────────────
        self.aspp = ASPP(
            in_channels  = 2048,
            out_channels = 256,
            dilations    = aspp_dilations,
            dropout      = dropout,
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        # Reduce low-level features (256ch from layer1) to 48ch
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Fuse: 256 (ASPP) + 48 (low-level) → 256 → 256 → num_classes
        self.decoder = nn.Sequential(
            AtrousSeparableConv(256 + 48, 256),
            nn.Dropout2d(dropout),
            AtrousSeparableConv(256, 256),
            nn.Conv2d(256, num_classes, 1),
        )

        self._init_new_layers()

    @staticmethod
    def _make_dilated(layer, stride, dilation):
        """Replace stride-2 with dilation to preserve spatial resolution."""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride  = (stride, stride)
                if module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding  = (dilation, dilation)

    def _init_new_layers(self):
        """Initialise non-backbone layers."""
        for module in [self.aspp, self.low_level_proj, self.decoder]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias,   0)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # Backbone
        x      = self.layer0(x)              # 1/4
        low    = self.layer1(x)              # 1/4, 256ch  ← low-level features
        x      = self.layer2(low)            # 1/8
        x      = self.layer3(x)             # 1/16
        x      = self.layer4(x)             # 1/16, 2048ch

        # ASPP
        aspp_out = self.aspp(x)             # 1/16, 256ch

        # Decoder
        # 1. Upsample ASPP to low-level feature resolution (1/4)
        aspp_up = F.interpolate(
            aspp_out, size=low.shape[2:],
            mode="bilinear", align_corners=False
        )
        # 2. Project low-level features
        low_proj = self.low_level_proj(low)  # 1/4, 48ch

        # 3. Concatenate and decode
        fused   = torch.cat([aspp_up, low_proj], dim=1)  # 1/4, 304ch
        out     = self.decoder(fused)                     # 1/4, num_classes

        # 4. Bilinear upsample to full resolution
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return out

    def freeze_backbone(self):
        """Freeze backbone weights (use when fine-tuning head only)."""
        for param in self.layer0.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters (stage 2 of training)."""
        for param in self.parameters():
            param.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_deeplabv3plus(config):
    return DeepLabV3Plus(
        in_channels        = config["in_channels"],
        num_classes        = config["num_classes"],
        output_stride      = config.get("output_stride", 16),
        aspp_dilations     = tuple(config.get("aspp_dilations", [6, 12, 18])),
        dropout            = config.get("dropout", 0.1),
        pretrained_backbone= config.get("pretrained_backbone", False),
    )


if __name__ == "__main__":
    model = DeepLabV3Plus(in_channels=11, num_classes=4)
    x     = torch.randn(2, 11, 256, 256)
    out   = model(x)
    print(f"DeepLabv3+ output: {out.shape}")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DeepLabv3+ trainable parameters: {params:,}")
