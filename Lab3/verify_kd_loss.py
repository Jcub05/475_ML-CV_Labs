"""
Verify that KD loss calculation produces comparable scales for training and validation
"""
import torch
import torch.nn.functional as F
import sys

def response_based_kd_loss_OLD(student_logits, teacher_logits, targets, temperature=3.0, alpha=0.5, beta=0.5):
    """OLD version with batchmean"""
    ce_loss = F.cross_entropy(student_logits, targets, ignore_index=255)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    total_loss = alpha * ce_loss + beta * kd_loss
    return total_loss, ce_loss.item(), kd_loss.item()

def response_based_kd_loss_NEW(student_logits, teacher_logits, targets, temperature=3.0, alpha=0.5, beta=0.5):
    """NEW version with manual normalization to match CE loss scale"""
    ce_loss = F.cross_entropy(student_logits, targets, ignore_index=255)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    
    # Use batchmean but normalize by spatial dimensions (H*W) to match CE loss scale
    # CE loss averages over (B, H, W), so KD should too
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
    # Normalize by spatial dimensions to match cross_entropy behavior
    spatial_size = student_logits.shape[2] * student_logits.shape[3]  # H * W
    kd_loss = kd_loss / spatial_size * (temperature ** 2)
    
    total_loss = alpha * ce_loss + beta * kd_loss
    return total_loss, ce_loss.item(), kd_loss.item()

def test_kd_loss_scales():
    """Test that KD loss is on comparable scale to CE loss"""
    print("="*70)
    print("VERIFYING KNOWLEDGE DISTILLATION LOSS SCALES")
    print("="*70)
    
    # Create dummy data similar to actual training
    batch_size = 8
    num_classes = 21
    height, width = 256, 256
    
    # Random student and teacher outputs
    torch.manual_seed(42)
    student_logits = torch.randn(batch_size, num_classes, height, width)
    teacher_logits = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    print(f"\nInput shapes:")
    print(f"  Student logits: {student_logits.shape}")
    print(f"  Teacher logits: {teacher_logits.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Total spatial elements: {batch_size * height * width:,}")
    
    # Test OLD version
    print("\n" + "="*70)
    print("OLD VERSION (reduction='batchmean'):")
    print("="*70)
    total_old, ce_old, kd_old = response_based_kd_loss_OLD(
        student_logits, teacher_logits, targets, 
        temperature=3.0, alpha=0.5, beta=0.5
    )
    print(f"  CE Loss:       {ce_old:.6f}")
    print(f"  KD Loss:       {kd_old:.6f}")
    print(f"  Total Loss:    {total_old:.6f}")
    print(f"  KD/CE Ratio:   {kd_old/ce_old:.1f}x")
    
    # Test NEW version
    print("\n" + "="*70)
    print("NEW VERSION (reduction='mean'):")
    print("="*70)
    total_new, ce_new, kd_new = response_based_kd_loss_NEW(
        student_logits, teacher_logits, targets, 
        temperature=3.0, alpha=0.5, beta=0.5
    )
    print(f"  CE Loss:       {ce_new:.6f}")
    print(f"  KD Loss:       {kd_new:.6f}")
    print(f"  Total Loss:    {total_new:.6f}")
    print(f"  KD/CE Ratio:   {kd_new/ce_new:.1f}x")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON:")
    print("="*70)
    print(f"  OLD KD loss was {kd_old/kd_new:.1f}x larger than NEW")
    print(f"  OLD total loss was {total_old/total_new:.1f}x larger than NEW")
    
    # Check if NEW version has comparable scales
    print("\n" + "="*70)
    print("VERIFICATION RESULTS:")
    print("="*70)
    
    if 0.1 < kd_new/ce_new < 10:
        print("  ✓ PASS: KD loss and CE loss are on comparable scales")
        print(f"    (ratio = {kd_new/ce_new:.2f}x, should be between 0.1x and 10x)")
    else:
        print("  ✗ FAIL: KD loss and CE loss scales are too different")
        print(f"    (ratio = {kd_new/ce_new:.2f}x, should be between 0.1x and 10x)")
        return False
    
    if 0.1 < total_new < 10:
        print(f"  ✓ PASS: Total loss is reasonable ({total_new:.4f})")
        print("    (similar to CE loss scale, suitable for training)")
    else:
        print(f"  ✗ FAIL: Total loss seems unreasonable ({total_new:.4f})")
        return False
    
    print("\n" + "="*70)
    print("EXPECTED BEHAVIOR IN TRAINING:")
    print("="*70)
    print("  • Training loss should be ~0.2 to ~2.0 (comparable to 'none' mode)")
    print("  • Validation loss should be similar scale to training loss")
    print("  • Both losses should appear clearly on the same plot")
    print("  • No more flat validation loss line near zero!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = test_kd_loss_scales()
    sys.exit(0 if success else 1)
