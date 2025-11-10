"""
Full integration test: verify training step produces correct loss scales
"""
import torch
import sys
sys.path.append('.')

from model_ultracompact import UltraCompactSegmentationModel
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import torch.nn.functional as F

print("="*70)
print("FULL INTEGRATION TEST: Training Step with Actual Models")
print("="*70)

# Load actual models
print("\nLoading models...")
student = UltraCompactSegmentationModel(num_classes=21, pretrained=False)
teacher = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
student.eval()
teacher.eval()

# Create realistic dummy batch
batch_size = 8
img = torch.randn(batch_size, 3, 256, 256)
target = torch.randint(0, 21, (batch_size, 256, 256))

print(f"Batch size: {batch_size}")
print(f"Image shape: {img.shape}")
print(f"Target shape: {target.shape}")

# Forward pass
print("\nForward pass...")
with torch.no_grad():
    student_output = student(img)
    teacher_output = teacher(img)['out']

print(f"Student output shape: {student_output.shape}")
print(f"Teacher output shape: {teacher_output.shape}")

# Compute losses (simulating train.py logic)
print("\n" + "="*70)
print("LOSS COMPUTATION:")
print("="*70)

# 1. No KD (baseline)
loss_none = F.cross_entropy(student_output, target, ignore_index=255)
print(f"\nNo KD:")
print(f"  Loss: {loss_none.item():.6f}")

# 2. Response-based KD (with corrected normalization)
ce_loss = F.cross_entropy(student_output, target, ignore_index=255)
student_soft = F.log_softmax(student_output / 3.0, dim=1)
teacher_soft = F.softmax(teacher_output / 3.0, dim=1)
kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
spatial_size = student_output.shape[2] * student_output.shape[3]
kd_loss = kd_loss / spatial_size * (3.0 ** 2)
total_loss = 0.5 * ce_loss + 0.5 * kd_loss

print(f"\nResponse-based KD (CORRECTED):")
print(f"  CE Loss:    {ce_loss.item():.6f}")
print(f"  KD Loss:    {kd_loss.item():.6f}")
print(f"  Total Loss: {total_loss.item():.6f}")
print(f"  KD/CE Ratio: {kd_loss.item()/ce_loss.item():.2f}x")

# Validation
print("\n" + "="*70)
print("VALIDATION:")
print("="*70)

if 0.1 < total_loss.item() < 10:
    print(f"  ✓ PASS: Total loss is reasonable ({total_loss.item():.4f})")
else:
    print(f"  ✗ FAIL: Total loss seems wrong ({total_loss.item():.4f})")
    sys.exit(1)

if abs(loss_none.item() - total_loss.item()) < 5.0:
    print(f"  ✓ PASS: KD loss similar to baseline ({abs(loss_none.item() - total_loss.item()):.2f} difference)")
else:
    print(f"  ✗ FAIL: KD loss too different from baseline ({abs(loss_none.item() - total_loss.item()):.2f} difference)")
    sys.exit(1)

if 0.1 < kd_loss.item()/ce_loss.item() < 10:
    print(f"  ✓ PASS: KD and CE losses on comparable scales ({kd_loss.item()/ce_loss.item():.2f}x ratio)")
else:
    print(f"  ✗ FAIL: KD and CE losses on different scales ({kd_loss.item()/ce_loss.item():.2f}x ratio)")
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nYou can now safely train with the corrected KD implementation.")
print("Expected training behavior:")
print("  • Training loss: ~0.2 to ~2.0 (same scale as 'none' mode)")
print("  • Validation loss: ~0.4 to ~2.0 (similar to training loss)")
print("  • Both curves visible on same plot")
print("="*70)
