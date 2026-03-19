"""
UNet Architecture for Retinal Vessel Segmentation
================================================
Standard U-Net implementation with skip connections.
Optimized for DRIVE dataset (3-channel input, 1-channel binary output).

Architecture summary:
- Encoder: 4 levels [64→128→256→512] with max pooling
- Bottleneck: 1024 channels  
- Decoder: 4 levels with transposed convolutions + skip connections
- Final 1×1 conv: produces probability map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv3x3 → BatchNorm → ReLU) × 2

    Maintains spatial dimensions via 'same' padding (kernel=3, padding=1).
    Used throughout encoder, bottleneck, and decoder.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.

    Args:
        in_channels: Input image channels (3 for RGB)
        out_channels: Output mask channels (1 for binary segmentation)
        features: Number of channels at each encoder level [64, 128, 256, 512]

    Forward pass produces logit map (apply sigmoid for probabilities).
    """

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder (contracting path)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck (deepest layer)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (expanding path) 
        for feature in reversed(features):
            # Transposed conv for upsampling
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # DoubleConv for feature refinement after skip connection
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Final 1×1 convolution to produce output channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder: downsampling with skip connection storage
        for enc_block in self.encoder:
            x = enc_block(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections to match decoder levels
        skip_connections = skip_connections[::-1]

        # Decoder: upsampling + skip connections
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Upsample
            skip = skip_connections[i // 2]

            # Concatenate skip connection along channel dimension
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.decoder[i + 1](concat_skip)  # Refine features

        # Final output layer (logits)
        return self.final_conv(x)


# Quick test to verify model loads correctly
if __name__ == "__main__":
    # Test instantiation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Test input: batch_size=2, channels=3, height=512, width=512
    x = torch.randn(2, 3, 512, 512).to(device)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        preds = model(x)
        print(f"Output shape: {preds.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("Model architecture verified ✓")
