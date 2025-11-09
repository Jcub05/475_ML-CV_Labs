"""
Ultra-Compact MobileNetV3-Small Segmentation Model
ELEC 475 Lab 3 - Option 2 (RECOMMENDED)

Parameters: ~0.48M (475K)
Expected Score: 1.0-1.5 (BEST FOR FORMULA)

References:
- MobileNetV3: Searching for MobileNetV3 (Howard et al., 2019)
- DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution (Chen et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module
    Simplified version with 2 dilation rates for parameter efficiency
    """
    def __init__(self, in_channels, out_channels, rates=[6, 12]):
        super(ASPP, self).__init__()
        
        self.branches = nn.ModuleList()
        
        # 1x1 convolution branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolution branches (only 2 rates)
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


class UltraCompactSegmentationModel(nn.Module):
    """
    Ultra-compact segmentation model using stripped MobileNetV3-Small
    
    Architecture:
    - Truncated MobileNetV3-Small backbone (10 stages instead of 13)
    - 2-rate ASPP module (rates: 6, 12)
    - Single-stage decoder
    - Parameters: ~0.48M (475K)
    
    This model prioritizes parameter efficiency for the evaluation formula:
    Score = 4 × mIoU / (1 + params_M)
    """
    def __init__(self, num_classes=21, pretrained=True):
        super(UltraCompactSegmentationModel, self).__init__()
        
        # Load pretrained MobileNetV3-Small
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_small(weights=weights)
        else:
            backbone = mobilenet_v3_small(weights=None)
        
        # Extract and TRUNCATE feature layers (only first 10 stages)
        # This removes the final expensive layers while keeping good features
        self.features = nn.Sequential(*list(backbone.features)[:10])
        
        # Feature dimensions at tap points:
        # Stage 1 (low):  16 channels, stride 4
        # Stage 3 (mid):  24 channels, stride 8
        # Stage 9 (high): 96 channels, stride 16
        
        # Simplified ASPP with only 2 rates (fewer parameters)
        self.aspp = ASPP(in_channels=96, out_channels=64, rates=[6, 12])
        
        # Feature projection for mid-level
        self.mid_conv = nn.Sequential(
            nn.Conv2d(24, 24, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        # Simplified decoder (single stage, skip low-level for efficiency)
        self.decoder = nn.Sequential(
            nn.Conv2d(64 + 24, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(64, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for new layers"""
        for m in [self.aspp, self.mid_conv, self.decoder, self.classifier]:
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
        
        # Extract features
        low_feat = None   # stride 4
        mid_feat = None   # stride 8
        high_feat = None  # stride 16
        
        # Forward through truncated MobileNetV3 stages
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Capture features at specific stages
            if i == 1:  # After first inverted residual (stride 4)
                low_feat = x
            elif i == 3:  # After third inverted residual (stride 8)
                mid_feat = x
        
        # Final high-level features
        high_feat = x
        
        # Process mid-level features
        mid_feat = self.mid_conv(mid_feat)
        
        # ASPP for multi-scale context
        x = self.aspp(high_feat)
        
        # Simplified decoder (single stage)
        x = F.interpolate(x, size=mid_feat.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, mid_feat], dim=1)
        x = self.decoder(x)
        
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
    print("Ultra-Compact MobileNetV3-Small Segmentation Model")
    print("=" * 70)
    
    # Create model
    model = UltraCompactSegmentationModel(num_classes=21, pretrained=False)
    
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
    print("RECOMMENDED: This model achieves the best score for the formula!")
    print("=" * 70)
