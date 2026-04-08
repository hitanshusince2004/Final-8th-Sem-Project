"""
Glacier Melting Detection — CNN Model
======================================
Convolutional Neural Network for multi-spectral glacier classification.

Architecture:
  Input (B, 11, 256, 256)
  -> 4x ConvBlock (Conv-BN-ReLU-Conv-BN-ReLU + MaxPool)
  -> GlobalAveragePool + Flatten
  -> FC(512) -> Dropout -> FC(256) -> Dropout -> FC(num_classes)

Also includes a pixel-wise CNN variant (fully-convolutional) that
outputs (B, C, H, W) segmentation logits for patch-level predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU (standard conv block)."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """
    Residual block: two ConvBNReLU + skip connection.
    If in_ch != out_ch, uses a 1x1 projection on the shortcut.
    """

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1   = ConvBNReLU(in_ch,  out_ch)
        self.conv2   = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                         if in_ch != out_ch else nn.Identity())
        self.relu     = nn.ReLU(inplace=True)
        self.dropout  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = self.dropout(out)
        return self.relu(out + self.shortcut(x))


# ══════════════════════════════════════════════════════════════════════════════
# CNN CLASSIFIER (image-level)
# ══════════════════════════════════════════════════════════════════════════════

class GlacierCNN(nn.Module):
    """
    CNN for patch-level multi-class glacier classification.
    Predicts a single class label per 256x256 patch.

    Suitable for: coarse glacier / non-glacier patch classification,
    image-level feature extraction for downstream tasks.
    """

    def __init__(self, in_channels=11, num_classes=4,
                 base_filters=32, dropout=0.3):
        super().__init__()
        f = base_filters

        # Encoder: 4 residual stages with progressive downsampling
        self.stage1 = nn.Sequential(
            ResidualConvBlock(in_channels, f,     dropout=0.0),
            nn.MaxPool2d(2),        # 256 -> 128
        )
        self.stage2 = nn.Sequential(
            ResidualConvBlock(f,     f*2,  dropout=0.0),
            nn.MaxPool2d(2),        # 128 -> 64
        )
        self.stage3 = nn.Sequential(
            ResidualConvBlock(f*2,   f*4,  dropout=dropout/2),
            nn.MaxPool2d(2),        # 64 -> 32
        )
        self.stage4 = nn.Sequential(
            ResidualConvBlock(f*4,   f*8,  dropout=dropout/2),
            nn.MaxPool2d(2),        # 32 -> 16
        )
        self.stage5 = nn.Sequential(
            ResidualConvBlock(f*8,   f*16, dropout=dropout),
            nn.AdaptiveAvgPool2d(4),     # -> (B, f*16, 4, 4)
        )

        flat_dim = f * 16 * 4 * 4

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.classifier(x)

    def extract_features(self, x):
        """Return feature map before the classifier head."""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x.flatten(1)


# ══════════════════════════════════════════════════════════════════════════════
# FULLY-CONVOLUTIONAL CNN (pixel-level segmentation)
# ══════════════════════════════════════════════════════════════════════════════

class GlacierFCN(nn.Module):
    """
    Fully Convolutional Network for pixel-wise glacier segmentation.
    Lighter than U-Net - no skip connections, faster to train.
    Encoder-only with ASPP-lite for multi-scale context.

    Input:  (B, 11, 256, 256)
    Output: (B, num_classes, 256, 256)
    """

    def __init__(self, in_channels=11, num_classes=4,
                 base_filters=32, dropout=0.3):
        super().__init__()
        f = base_filters

        self.encoder = nn.Sequential(
            ResidualConvBlock(in_channels, f),
            nn.MaxPool2d(2),
            ResidualConvBlock(f,     f*2),
            nn.MaxPool2d(2),
            ResidualConvBlock(f*2,   f*4, dropout=dropout/2),
            nn.MaxPool2d(2),
            ResidualConvBlock(f*4,   f*8, dropout=dropout),
        )

        # ASPP-lite: parallel dilated convolutions
        self.aspp1 = nn.Conv2d(f*8, f*4, 1, bias=False)
        self.aspp2 = nn.Conv2d(f*8, f*4, 3, padding=3,  dilation=3,  bias=False)
        self.aspp3 = nn.Conv2d(f*8, f*4, 3, padding=6,  dilation=6,  bias=False)
        self.aspp4 = nn.Conv2d(f*8, f*4, 3, padding=12, dilation=12, bias=False)
        self.aspp_bn = nn.BatchNorm2d(f*16)
        self.aspp_relu = nn.ReLU(inplace=True)

        self.decoder = nn.Sequential(
            nn.Conv2d(f*16, f*8, 3, padding=1, bias=False),
            nn.BatchNorm2d(f*8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(f*8, f*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(f*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(f*4, num_classes, 1),
        )

    def forward(self, x):
        h, w  = x.shape[2], x.shape[3]
        feat  = self.encoder(x)

        # ASPP
        a1 = self.aspp_relu(self.aspp1(feat))
        a2 = self.aspp_relu(self.aspp2(feat))
        a3 = self.aspp_relu(self.aspp3(feat))
        a4 = self.aspp_relu(self.aspp4(feat))
        aspp_out = self.aspp_bn(torch.cat([a1, a2, a3, a4], dim=1))

        out = self.decoder(aspp_out)

        # Upsample to original resolution
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


class GlacierResNet(nn.Module):
    """
    ResNet-based CNN for patch-level classification.
    Supports multi-spectral input (11 channels).
    """
    def __init__(self, in_channels=11, num_classes=4, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet18":
            self.model = models.resnet18(weights=None)
        elif backbone == "resnet34":
            self.model = models.resnet34(weights=None)
        else:
            self.model = models.resnet18(weights=None)
            
        # Adapt first layer for multi-spectral input
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Adapt last layer for our number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_cnn(config, segmentation=True, model_type="fcn"):
    """
    Build CNN model from config dict.
    segmentation=True:
        model_type="fcn" -> GlacierFCN
    segmentation=False:
        model_type="basic" -> GlacierCNN
        model_type="resnet18" -> GlacierResNet(resnet18)
        model_type="resnet34" -> GlacierResNet(resnet34)
        model_type="resnet50" -> GlacierResNet(resnet50)
    """
    if segmentation:
        return GlacierFCN(
            in_channels = config["in_channels"],
            num_classes  = config["num_classes"],
            base_filters = config.get("base_filters", 32),
            dropout      = config.get("dropout", 0.3),
        )
    else:
        if model_type == "basic":
            return GlacierCNN(
                in_channels = config["in_channels"],
                num_classes  = config["num_classes"],
                base_filters = config.get("base_filters", 32),
                dropout      = config.get("dropout", 0.3),
            )
        else:
            return GlacierResNet(
                in_channels = config["in_channels"],
                num_classes = config["num_classes"],
                backbone = model_type if model_type in ["resnet18", "resnet34"] else "resnet18"
            )


if __name__ == "__main__":
    # Quick architecture test
    model = GlacierFCN(in_channels=11, num_classes=4)
    x     = torch.randn(2, 11, 256, 256)
    out   = model(x)
    print(f"GlacierFCN output: {out.shape}")      # (2, 4, 256, 256)

    model2 = GlacierCNN(in_channels=11, num_classes=4)
    out2   = model2(x)
    print(f"GlacierCNN output: {out2.shape}")     # (2, 4)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FCN parameters: {total:,}")
