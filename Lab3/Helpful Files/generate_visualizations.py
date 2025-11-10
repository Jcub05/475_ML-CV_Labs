"""
Generate visualizations for trained models (best and worst predictions)
Standalone script - doesn't re-run evaluation
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Import models and utilities
from model_ultracompact import UltraCompactSegmentationModel
from model_ultracompact_v2 import UltraCompactSegmentationModelV2
from model_standard import StandardSegmentationModel
from train import VOCNormalize, collate_fn

# VOC color map
VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
    'horse', 'motorbike', 'person', 'potted plant', 'sheep',
    'sofa', 'train', 'tv/monitor'
]


def decode_segmap(label_mask, num_classes=21):
    """Decode segmentation mask to RGB"""
    r = np.zeros_like(label_mask, dtype=np.uint8)
    g = np.zeros_like(label_mask, dtype=np.uint8)
    b = np.zeros_like(label_mask, dtype=np.uint8)
    
    for class_idx in range(num_classes):
        idx = label_mask == class_idx
        r[idx] = VOC_COLORMAP[class_idx][0]
        g[idx] = VOC_COLORMAP[class_idx][1]
        b[idx] = VOC_COLORMAP[class_idx][2]
    
    return np.stack([r, g, b], axis=2)


def calculate_miou(pred, target, num_classes=21):
    """Calculate mIoU"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            continue
        
        iou = intersection / union
        ious.append(iou)
    
    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
    per_class_iou = {cls: ious[cls] if cls < len(ious) else None for cls in range(num_classes)}
    return mean_iou, per_class_iou


def generate_visualizations(model, dataset, device, num_samples=10, save_dir='./visualizations'):
    """Generate best and worst prediction visualizations"""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Evaluating all samples to find best and worst predictions...")
    all_ious = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Computing IoUs'):
            img, target = dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)
            h, w = target.shape
            
            output = model(img_tensor)
            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
            pred = output.squeeze(0).argmax(0).cpu()
            
            miou, _ = calculate_miou(pred, target)
            all_ious.append((idx, miou))
    
    # Sort by IoU
    all_ious.sort(key=lambda x: x[1], reverse=True)
    
    # Select top and bottom samples
    num_best = num_samples // 2
    num_worst = num_samples - num_best
    
    best_indices = [idx for idx, _ in all_ious[:num_best]]
    worst_indices = [idx for idx, _ in all_ious[-num_worst:]]
    indices = best_indices + worst_indices
    
    print(f"Selected {num_best} best predictions (IoU: {all_ious[0][1]:.4f} - {all_ious[num_best-1][1]:.4f})")
    print(f"Selected {num_worst} worst predictions (IoU: {all_ious[-1][1]:.4f} - {all_ious[-num_worst][1]:.4f})")
    
    # Generate visualizations
    visualization_results = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc='Generating visualizations'):
            img, target = dataset[idx]
            
            # Get original image
            img_path = dataset.images[idx]
            original_img = np.array(Image.open(img_path).convert('RGB'))
            
            # Forward pass
            img_tensor = img.unsqueeze(0).to(device)
            h, w = target.shape
            
            output = model(img_tensor)
            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
            pred = output.squeeze(0).argmax(0).cpu()
            
            # Calculate IoU
            miou, _ = calculate_miou(pred, target)
            
            # Track for summary
            quality = 'Best' if idx in best_indices else 'Worst'
            visualization_results.append({
                'index': idx,
                'filename': f'visualization_{quality.lower()}_{idx}.png',
                'miou': miou,
                'quality': quality
            })
            
            # Decode masks
            gt_color = decode_segmap(target.cpu().numpy())
            pred_color = decode_segmap(pred.numpy())
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(gt_color)
            axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(pred_color)
            axes[2].set_title(f'Prediction (IoU: {miou:.4f})', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            plt.suptitle(f'{quality} Example - Image {idx}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_dir / f'visualization_{quality.lower()}_{idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # Save summary
    summary_path = save_dir / 'visualization_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VISUALIZATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("BEST PREDICTIONS:\n")
        f.write("-" * 80 + "\n")
        for result in visualization_results:
            if result['quality'] == 'Best':
                f.write(f"  {result['filename']:40s}  mIoU: {result['miou']:.4f}\n")
        
        f.write("\nWORST PREDICTIONS:\n")
        f.write("-" * 80 + "\n")
        for result in visualization_results:
            if result['quality'] == 'Worst':
                f.write(f"  {result['filename']:40s}  mIoU: {result['miou']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Total visualizations: {len(visualization_results)}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✓ Visualizations saved to {save_dir}/")
    print(f"✓ Summary saved to {summary_path}")


def create_legend(save_dir):
    """Create color legend for classes"""
    save_dir = Path(save_dir)
    
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')
    
    # Create color patches
    for idx, (class_name, color) in enumerate(zip(VOC_CLASSES, VOC_COLORMAP)):
        color_normalized = [c / 255.0 for c in color]
        ax.add_patch(plt.Rectangle((0, idx * 0.5), 0.5, 0.4, facecolor=color_normalized))
        ax.text(0.6, idx * 0.5 + 0.2, class_name, fontsize=12, va='center')
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(VOC_CLASSES) * 0.5)
    ax.invert_yaxis()
    
    plt.title('Class Color Legend', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_dir / 'class_legend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Legend saved to {save_dir / 'class_legend.png'}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for trained models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, required=True, choices=['ultracompact', 'ultracompact_v2', 'standard'],
                        help='Model type')
    parser.add_argument('--num_vis', type=int, default=10, help='Number of visualizations (split best/worst)')
    parser.add_argument('--save_dir', type=str, default='./visualizations', help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\n{'='*80}")
    print("Generating Visualizations")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print(f"Number of samples: {args.num_vis}")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    if args.model == 'ultracompact':
        model = UltraCompactSegmentationModel(num_classes=21, pretrained=False)
    elif args.model == 'ultracompact_v2':
        model = UltraCompactSegmentationModelV2(num_classes=21, pretrained=False)
    elif args.model == 'standard':
        model = StandardSegmentationModel(num_classes=21, pretrained=False)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Load dataset
    print("\nLoading validation dataset...")
    val_transform = VOCNormalize(augment=False)
    
    # Try multiple dataset locations
    dataset_roots = [
        './data',
        '../data',
        '../../data',
        str(Path.home() / '.cache' / 'kagglehub' / 'datasets' / 'huanghanchina' / 'pascal-voc-2012' / 'versions' / '1'),
    ]
    
    val_dataset = None
    for root in dataset_roots:
        try:
            val_dataset = VOCSegmentation(
                root=root,
                year='2012',
                image_set='val',
                download=False,
                transforms=val_transform
            )
            print(f"✓ Dataset loaded from: {root}")
            break
        except:
            continue
    
    if val_dataset is None:
        raise RuntimeError("Could not find dataset!")
    
    print(f"Validation set size: {len(val_dataset)}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(model, val_dataset, device, num_samples=args.num_vis, save_dir=args.save_dir)
    
    # Create legend
    create_legend(args.save_dir)
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
