"""
Glacier Melting Detection — U-Net Segmentation Model
=====================================================
Full U-Net implementation for glacier semantic segmentation.

Architecture (depth=4):
  Encoder:
    Input (B, 11, 256, 256)
    -> DoubleConv(64) -> MaxPool -> (B, 64, 128, 128)
    -> DoubleConv(128) -> MaxPool -> (B, 128, 64, 64)
    -> DoubleConv(256) -> MaxPool -> (B, 256, 32, 32)
    -> DoubleConv(512) -> MaxPool -> (B, 512, 16, 16)
  Bottleneck:
    -> DoubleConv(1024) -> (B, 1024, 16, 16)
  Decoder (with skip connections):
    -> Up + concat(skip) -> DoubleConv(512)
    -> Up + concat(skip) -> DoubleConv(256)
    -> Up + concat(skip) -> DoubleConv(128)
    -> Up + concat(skip) -> DoubleConv(64)
  Output:
    -> Conv1x1(num_classes) -> (B, num_classes, 256, 256)

Reference: Ronneberger et al. 2015 - U-Net: CNNs for Biomedical Image Segmentation
Adapted for multi-spectral satellite imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """
    (Conv -> BN -> ReLU) x 2 - the fundamental U-Net block.
    Optional dropout between the two convolutions (for regularisation).
    """

    def __init__(self, in_ch, out_ch, mid_ch=None, dropout=0.0):
        super().__init__()
        mid = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d(2) → DoubleConv — encoder step."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Decoder step: upsample + concatenate skip + DoubleConv.
    bilinear=True uses bilinear upsampling (lighter);
    bilinear=False uses transposed convolution (learnable).
    """

    def __init__(self, in_ch, out_ch, bilinear=True, dropout=0.0):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch//2, dropout=dropout)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x1, x2):
        """
        x1: decoder feature map (to upsample)
        x2: encoder skip-connection feature map
        """
        x1 = self.up(x1)

        # Pad x1 to match x2 if sizes differ (due to odd input dimensions)
        dh = x2.shape[2] - x1.shape[2]
        dw = x2.shape[3] - x1.shape[3]
        if dh > 0 or dw > 0:
            x1 = F.pad(x1, [dw//2, dw - dw//2, dh//2, dh - dh//2])

        return self.conv(torch.cat([x2, x1], dim=1))


class AttentionGate(nn.Module):
    """
    Attention Gate - Schlemper et al. 2019.
    Helps focus on glacier regions by learning which skip-connection
    features are relevant at each decoder stage.
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """g = gating signal (from decoder), x = skip connection (from encoder)."""
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample g1 to match x1 if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi   # attended skip features


# ══════════════════════════════════════════════════════════════════════════════
# U-NET
# ══════════════════════════════════════════════════════════════════════════════

class UNet(nn.Module):
    """
    U-Net for multi-spectral glacier segmentation.

    Args:
        in_channels:  number of input bands (default: 11)
        num_classes:  number of output classes (default: 4)
        base_filters: number of filters in the first encoder block (default: 64)
        depth:        number of encoder/decoder stages (default: 4)
        bilinear:     use bilinear upsampling instead of transposed conv (default: True)
        dropout:      dropout rate in decoder blocks (default: 0.2)
        attention:    use attention gates in decoder (default: True)
    """

    def __init__(self, in_channels=11, num_classes=4,
                 base_filters=64, depth=4,
                 bilinear=True, dropout=0.2, attention=True):
        super().__init__()
        self.in_channels  = in_channels
        self.num_classes  = num_classes
        self.depth        = depth
        self.attention    = attention

        # Build encoder dynamically based on depth
        f = base_filters
        self.inc   = DoubleConv(in_channels, f)           # initial: no pool
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.att_gates = nn.ModuleList() if attention else None

        in_f = f
        filters = [f]
        for i in range(depth):
            out_f = in_f * 2
            drop  = dropout if i >= depth // 2 else 0.0
            self.downs.append(Down(in_f, out_f, dropout=drop))
            filters.append(out_f)
            in_f = out_f

        # Bottleneck (no extra pool - already at 1/2^depth resolution)
        self.bottleneck = DoubleConv(in_f, in_f * 2, dropout=dropout)
        bottleneck_f = in_f * 2

        # Build decoder in reverse
        dec_in = bottleneck_f
        for i in range(depth - 1, -1, -1):
            skip_f = filters[i + 1]
            out_f  = skip_f
            if attention:
                self.att_gates.append(
                    AttentionGate(F_g=dec_in, F_l=skip_f, F_int=skip_f // 2)
                )
            self.ups.append(Up(dec_in + skip_f, out_f, bilinear=bilinear, dropout=dropout))
            dec_in = out_f

        # Final 1×1 conv
        self.out_conv = nn.Conv2d(filters[1], num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x):
        # Encoder
        skips = []
        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, up in enumerate(self.ups):
            # skips has (depth + 1) elements: inc, down0, ..., down(depth-1)
            # We want to skip the last one (it's the bottleneck input)
            skip = skips[-(i + 2)]
            if self.attention:
                skip = self.att_gates[i](g=x, x=skip)
            x = up(x, skip)

        return self.out_conv(x)

    def get_intermediate_features(self, x):
        """Return encoder feature maps at each scale (for visualization)."""
        features = []
        x = self.inc(x)
        features.append(x)
        for down in self.downs:
            x = down(x)
            features.append(x)
        return features


# ══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT U-NET (for resource-constrained environments)
# ══════════════════════════════════════════════════════════════════════════════

class UNetLite(nn.Module):
    """
    Compact U-Net with configurable base filters.
    """

    def __init__(self, in_channels=11, num_classes=4, base_filters=16):
        super().__init__()
        f = base_filters
        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = Down(f,   f*2)
        self.enc3 = Down(f*2, f*4)
        # Bottleneck
        self.bot  = Down(f*4, f*8)
        # Decoder
        self.dec3 = Up(f*8 + f*4, f*4, bilinear=True)
        self.dec2 = Up(f*4 + f*2, f*2, bilinear=True)
        self.dec1 = Up(f*2 + f,   f,   bilinear=True)
        self.out  = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bot(e3)
        d3 = self.dec3(b,  e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out(d1)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_unet(config):
    return UNet(
        in_channels  = config["in_channels"],
        num_classes  = config["num_classes"],
        base_filters = config.get("base_filters", 64),
        depth        = config.get("depth", 4),
        bilinear     = config.get("bilinear", True),
        dropout      = config.get("dropout", 0.2),
        attention    = config.get("attention", True),
    )


if __name__ == "__main__":
    model = UNet(in_channels=11, num_classes=4, base_filters=64)
    x     = torch.randn(2, 11, 256, 256)
    out   = model(x)
    print(f"U-Net output: {out.shape}")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net trainable parameters: {params:,}")

    lite  = UNetLite(in_channels=11, num_classes=4)
    out2  = lite(x)
    print(f"UNet-Lite output: {out2.shape}")
    p2    = sum(p.numel() for p in lite.parameters() if p.requires_grad)
    print(f"UNet-Lite trainable parameters: {p2:,}")
