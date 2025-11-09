"""
Model Comparison Utility
ELEC 475 Lab 3

Provides factory function to get models and comparison utilities.
"""

import torch
from model_ultracompact import UltraCompactSegmentationModel
from model_standard import StandardSegmentationModel


def get_model(model_type, num_classes=21, pretrained=True):
    """
    Factory function to get model by type
    
    Args:
        model_type: 'ultracompact' or 'standard'
        num_classes: number of output classes
        pretrained: whether to use pretrained backbone
    
    Returns:
        model instance
    """
    if model_type == 'ultracompact':
        return UltraCompactSegmentationModel(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'standard':
        return StandardSegmentationModel(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'ultracompact' or 'standard'.")


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models():
    """Print comparison of both models"""
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Create models
    ultracompact = get_model('ultracompact', pretrained=False)
    standard = get_model('standard', pretrained=False)
    
    # Count parameters
    ultracompact_params = count_parameters(ultracompact)
    standard_params = count_parameters(standard)
    
    print(f"\nUltra-Compact Model:")
    print(f"  Parameters: {ultracompact_params:,} ({ultracompact_params/1e6:.2f}M)")
    
    print(f"\nStandard Model:")
    print(f"  Parameters: {standard_params:,} ({standard_params/1e6:.2f}M)")
    
    print(f"\nRatio: {standard_params/ultracompact_params:.2f}x")
    
    # Score predictions
    print("\n" + "=" * 80)
    print("SCORE PREDICTIONS (Score = 4 × mIoU / (1 + params_M))")
    print("=" * 80)
    
    print("\nUltra-Compact Model:")
    for miou in [0.35, 0.40, 0.45, 0.50]:
        score = 4 * miou / (1 + ultracompact_params/1e6)
        print(f"  mIoU={miou:.2f} → Score={score:.3f}")
    
    print("\nStandard Model:")
    for miou in [0.45, 0.50, 0.55, 0.60]:
        score = 4 * miou / (1 + standard_params/1e6)
        print(f"  mIoU={miou:.2f} → Score={score:.3f}")
    
    print("\n" + "=" * 80)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        out_uc = ultracompact(dummy_input)
        out_std = standard(dummy_input)
    
    print(f"  Ultra-Compact output: {out_uc.shape}")
    print(f"  Standard output: {out_std.shape}")
    print("\n[OK] Both models work correctly!")
    print("=" * 80)


if __name__ == '__main__':
    compare_models()
