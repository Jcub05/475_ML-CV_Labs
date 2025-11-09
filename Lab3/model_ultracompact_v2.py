"""
Ultra-Compact MobileNetV3-Small Segmentation Model V2
Improvements from V1:
- Extended backbone by one stage (uses first 11 feature blocks instead of 10)
- ASPP uses 3 dilation rates (6,12,18) instead of 2
- ASPP and decoder convolutional blocks use depthwise-separable convolutions
- Channel sizes dynamically determined by a dummy forward pass during __init__
- Maintains compatibility with knowledge distillation (return_features flag)

Expected: ~500-600k parameters, better mIoU than V1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# -----------------------
# Utility: Depthwise Separable Conv Block
# -----------------------
class SepConv(nn.Module):
    """
    Depthwise separable convolution block: Depthwise (spatial) -> Pointwise (1x1) -> BN -> Activation
    Can accept dilation for the depthwise conv.
    
    Benefits:
    - Far fewer parameters than regular conv (reduces by ~8-9x for 3x3 kernels)
    - Similar representational power when used carefully
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1, bias=False, activation=nn.ReLU):
        super(SepConv, self).__init__()
        # Depthwise conv (per-channel spatial filtering)
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation, groups=in_ch, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_ch)
        # Pointwise (1x1) conv to mix channels
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = activation(inplace=True) if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


# -----------------------
# ASPP Module (with separable convs)
# -----------------------
class ASPP(nn.Module):
    """
    Lightweight ASPP using depthwise-separable atrous branches.
    
    Branches:
      - 1x1 conv (pointwise)
      - SepConv (dilation=6)
      - SepConv (dilation=12)
      - SepConv (dilation=18)  <- NEW: Third rate for better multi-scale context
      - global avg pool -> 1x1 conv
    
    After concat -> 1x1 project -> BN -> ReLU -> Dropout
    """
    def __init__(self, in_channels, out_channels, rates=(6, 12, 18)):
        super(ASPP, self).__init__()
        self.branches = nn.ModuleList()

        # 1x1 conv branch (implemented as pointwise conv)
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Atrous separable conv branches (3 rates for better multi-scale)
        for rate in rates:
            padding = rate  # For dilation=rate, padding=rate maintains spatial size
            self.branches.append(SepConv(in_channels, out_channels, kernel_size=3, padding=padding, dilation=rate))

        # Global pooling branch
        self.branches.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Project concatenated features
        num_branches = len(self.branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        outs = []
        for i, branch in enumerate(self.branches):
            out = branch(x)
            # The last branch is global pool which we upsample to (h,w)
            if i == len(self.branches) - 1:
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        x = self.project(x)
        return x


# -----------------------
# Ultra-Compact Segmentation Model V2
# -----------------------
class UltraCompactSegmentationModelV2(nn.Module):
    """
    Revised ultra-compact segmentation model with improvements:
      - Extended backbone: keeps first 11 feature blocks from mobilenet_v3_small (was 10)
      - ASPP with rates [6,12,18] and separable convs (was [6,12] with regular convs)
      - Decoder uses separable convs (was regular convs)
      - Dynamically probes channel widths at init time using a dummy pass
      
    Architecture:
      Input (H, W, 3)
        ↓
      MobileNetV3-Small backbone (first 11 blocks, pretrained on ImageNet)
        ├─ low_feat  @ stride 4  (captured at block 1)
        ├─ mid_feat  @ stride 8  (captured at block 3)
        └─ high_feat @ stride 16 (output of block 10, now 11)
        ↓
      ASPP (3 dilation rates: 6, 12, 18) → 64 channels
        ↓
      Upsample 2x to stride 8
        ↓
      Concatenate with mid_feat (projected to 24 channels)
        ↓
      Decoder (2x SepConv blocks) → 64 channels
        ↓
      Upsample 8x to original resolution
        ↓
      Classifier (1x1 conv) → num_classes
    """
    def __init__(self, num_classes=21, pretrained=True, aspp_out=64, mid_proj_ch=24):
        super(UltraCompactSegmentationModelV2, self).__init__()
        
        # Load backbone (pretrained weights optional)
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_small(weights=weights)
        else:
            backbone = mobilenet_v3_small(weights=None)

        # Keep a truncated backbone: first 11 feature blocks (extended by one stage)
        # Using 11 rather than 10 to capture slightly deeper semantics
        self.features = nn.Sequential(*list(backbone.features)[:11])

        # Discover channel widths by running a dummy forward through self.features
        # This avoids hard-coded channel numbers and works across torchvision versions
        self._probe_feature_channels()

        # ASPP (use high_ch discovered, output 64 channels, 3 dilation rates)
        self.aspp = ASPP(in_channels=self.high_ch, out_channels=aspp_out, rates=(6, 12, 18))

        # Mid projection: take discovered mid_ch -> mid_proj_ch
        self.mid_conv = nn.Sequential(
            nn.Conv2d(self.mid_ch, mid_proj_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_proj_ch),
            nn.ReLU(inplace=True)
        )
        self.mid_proj_ch = mid_proj_ch

        # Decoder: concatenation of ASPP(out=aspp_out) upsampled to mid size + mid_proj_ch
        decoder_in_ch = aspp_out + mid_proj_ch
        # Use separable conv blocks in decoder (two small SepConv blocks for better mixing)
        self.decoder = nn.Sequential(
            SepConv(decoder_in_ch, 64, kernel_size=3, padding=1),
            SepConv(64, 64, kernel_size=3, padding=1),
        )

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        # Initialize new weights
        self._init_weights()

    def _probe_feature_channels(self):
        """
        Run a tiny dummy pass through self.features to find low/mid/high channel widths.
        Captures:
          - low_feat at i==1 (early, stride ~4)
          - mid_feat at i==3 (middle, stride ~8)
          - high_feat: final output of truncated features (stride ~16)
        """
        self.features.eval()
        with torch.no_grad():
            # dummy size matches typical input; smaller spatial dims save compute for init-time probe
            dummy = torch.zeros(1, 3, 224, 224)
            x = dummy
            low_feat = None
            mid_feat = None
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == 1:
                    low_feat = x
                if i == 3:
                    mid_feat = x
            high_feat = x

        # Save discovered channel widths
        # Fallbacks in case features were not captured — should not happen on standard mobilenet_v3_small
        self.low_ch = low_feat.shape[1] if low_feat is not None else 16
        self.mid_ch = mid_feat.shape[1] if mid_feat is not None else 24
        self.high_ch = high_feat.shape[1]

    def _init_weights(self):
        """Initialize weights for new layers only (not the pretrained backbone)."""
        for module in [self.aspp, self.mid_conv, self.decoder, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        """
        Forward pass with optional feature extraction for knowledge distillation.
        
        Args:
            x: Input tensor (B, 3, H, W)
            return_features: If True, returns (output, features_dict)
        
        Returns:
            output: Segmentation logits (B, num_classes, H, W)
            features: Dict with 'low', 'mid', 'high' features (if return_features=True)
        """
        input_shape = x.shape[-2:]
        low_feat = None
        mid_feat = None
        high_feat = None

        # Run through truncated backbone and capture taps
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 1:
                low_feat = x  # stride ~4
            elif i == 3:
                mid_feat = x  # stride ~8
        high_feat = x  # final truncated feature (stride ~16)

        # Project mid-level features
        mid_proj = self.mid_conv(mid_feat)

        # ASPP on high-level features
        x_aspp = self.aspp(high_feat)

        # Upsample ASPP output to mid spatial size and fuse
        x = F.interpolate(x_aspp, size=mid_proj.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, mid_proj], dim=1)

        # Decoder (separable convs)
        x = self.decoder(x)

        # Upsample to original resolution and classify
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        output = self.classifier(x)

        if return_features:
            features = {
                'low': low_feat,
                'mid': mid_feat,
                'high': high_feat
            }
            return output, features

        return output


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------
# Test and verification
# -----------------------
if __name__ == '__main__':
    print("=" * 70)
    print("Ultra-Compact MobileNetV3-Small Segmentation Model V2")
    print("=" * 70)

    # Create model
    model = UltraCompactSegmentationModelV2(num_classes=21, pretrained=False)

    # Count parameters
    params = count_parameters(model)
    print(f"\nTotal parameters: {params:,} ({params/1e6:.4f}M)")
    
    # Compare to V1
    print(f"\nComparison:")
    print(f"  V1 (original):  ~475,077 parameters (0.48M)")
    print(f"  V2 (improved):  {params:,} parameters ({params/1e6:.2f}M)")
    
    # Calculate expected score improvement needed
    v1_params_m = 0.48
    v2_params_m = params / 1e6
    print(f"\nFor same final score, V2 needs mIoU:")
    for v1_miou in [0.40, 0.45, 0.50]:
        v1_score = 4 * v1_miou / (1 + v1_params_m)
        # Solve: 4 * v2_miou / (1 + v2_params_m) = v1_score
        v2_miou_needed = v1_score * (1 + v2_params_m) / 4
        print(f"  If V1 gets {v1_miou:.2f} → V2 needs {v2_miou_needed:.3f} (Δ={v2_miou_needed-v1_miou:+.3f})")

    # Test forward pass
    print("\n" + "=" * 70)
    print("Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")

        # Test with feature extraction
        output, features = model(dummy_input, return_features=True)
        print(f"\n  Features for knowledge distillation:")
        for name, feat in features.items():
            print(f"    {name}: {feat.shape}")

    print("\n" + "=" * 70)
    print("✓ Model V2 ready for training!")
    print("=" * 70)
