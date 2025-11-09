"""
Standard MobileNetV3-Small Segmentation Model (Optimized)
ELEC 475 Lab 3 - Option 1

Parameters: ~1.36M (reduced from 3.41M)
Expected Score: 0.80-1.20

References:
- MobileNetV3: Searching for MobileNetV3 (Howard et al., 2019)
- DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution (Chen et al., 2018)
- ASPP: Rethinking Atrous Convolution for Semantic Image Segmentation (Chen et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module
    Captures multi-scale context using parallel atrous convolutions
    """
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        self.branches = nn.ModuleList()
        
        # 1x1 convolution branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolution branches
        for rate in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling branch
        self.branches.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Combine all branches
        num_branches = len(rates) + 2  # 1x1 + atrous branches + global pooling
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        h, w = x.shape[2:]
        branch_outputs = []
        
        for i, branch in enumerate(self.branches):
            out = branch(x)
            # Upsample global pooling branch to match spatial dimensions
            if i == len(self.branches) - 1:
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            branch_outputs.append(out)
        
        # Concatenate all branches
        x = torch.cat(branch_outputs, dim=1)
        x = self.project(x)
        
        return x


class StandardSegmentationModel(nn.Module):
    """
    Standard compact segmentation model using MobileNetV3-Small backbone (Optimized)
    
    Architecture:
    - Full MobileNetV3-Small backbone (pretrained)
    - Lightweight 2-rate ASPP module (rates: 6, 12) - reduced from 3 rates
    - Simplified decoder with reduced channels
    - Parameters: ~1.36M (reduced from 3.41M)
    """
    def __init__(self, num_classes=21, pretrained=True):
        super(StandardSegmentationModel, self).__init__()
        
        # Load pretrained MobileNetV3-Small
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_small(weights=weights)
        else:
            backbone = mobilenet_v3_small(weights=None)
        
        # Extract feature layers from MobileNetV3-Small
        self.features = backbone.features
        
        # Feature dimensions at tap points:
        # Stage 1 (low):  16 channels, stride 4
        # Stage 3 (mid):  24 channels, stride 8
        # Stage 12 (high): 576 channels, stride 16
        
        # Lightweight ASPP module (2 rates instead of 3, reduced output channels)
        self.aspp = ASPP(in_channels=576, out_channels=64, rates=[6, 12])
        
        # Feature projection layers (reduced channels)
        self.mid_conv = nn.Sequential(
            nn.Conv2d(24, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.low_conv = nn.Sequential(
            nn.Conv2d(16, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        # Simplified decoder with reduced channels
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(64 + 16, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(64 + 8, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(32, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for new layers"""
        for m in [self.aspp, self.mid_conv, self.low_conv, self.decoder_conv1, self.decoder_conv2, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: input image tensor (B, 3, H, W)
            return_features: if True, return intermediate features for knowledge distillation
            
        Returns:
            output: segmentation logits (B, num_classes, H, W)
            features (optional): dict of intermediate features for KD
        """
        input_shape = x.shape[-2:]
        
        # Extract multi-scale features from backbone
        low_feat = None   # stride 4
        mid_feat = None   # stride 8
        high_feat = None  # stride 16
        
        # Forward through MobileNetV3 stages
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Capture features at specific stages
            if i == 1:  # After first inverted residual block (stride 4)
                low_feat = x
            elif i == 3:  # After third inverted residual block (stride 8)
                mid_feat = x
        
        # Final high-level features (stride 16)
        high_feat = x
        
        # Process features
        mid_feat = self.mid_conv(mid_feat)
        low_feat = self.low_conv(low_feat)
        
        # ASPP for multi-scale context
        x = self.aspp(high_feat)
        
        # Decoder with skip connections
        # Upsample and concatenate with mid-level features
        x = F.interpolate(x, size=mid_feat.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, mid_feat], dim=1)
        x = self.decoder_conv1(x)
        
        # Upsample and concatenate with low-level features
        x = F.interpolate(x, size=low_feat.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_feat], dim=1)
        x = self.decoder_conv2(x)
        
        # Final upsampling to input resolution
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        # Classifier
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


if __name__ == '__main__':
    print("=" * 70)
    print("Standard MobileNetV3-Small Segmentation Model")
    print("=" * 70)
    
    # Create model
    model = StandardSegmentationModel(num_classes=21, pretrained=False)
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nTotal parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        
        # Test with feature extraction
        output, features = model(dummy_input, return_features=True)
        print(f"\n  Features for knowledge distillation:")
        for name, feat in features.items():
            print(f"    {name}: {feat.shape}")
    
    # Score estimation
    print("\n" + "=" * 70)
    print("Score Estimation: Score = 4 × mIoU / (1 + params_M)")
    print("=" * 70)
    
    for miou in [0.40, 0.45, 0.50, 0.55]:
        score = 4 * miou / (1 + params/1e6)
        print(f"  mIoU={miou:.2f} → Score={score:.3f}")
    
    print("\n" + "=" * 70)
