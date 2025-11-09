"""
Testing Script for Compact Segmentation Models
ELEC 475 Lab 3

Evaluates trained models on VOC2012 validation set and generates visualizations.
Calculates mIoU, inference speed, and final score.
"""

import argparse
import os
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm

# Import custom models
from model_ultracompact import UltraCompactSegmentationModel
from model_ultracompact_v2 import UltraCompactSegmentationModelV2
from model_standard import StandardSegmentationModel


# VOC color map for visualization
VOC_COLORMAP = [
    [0, 0, 0],          # 0: background
    [128, 0, 0],        # 1: aeroplane
    [0, 128, 0],        # 2: bicycle
    [128, 128, 0],      # 3: bird
    [0, 0, 128],        # 4: boat
    [128, 0, 128],      # 5: bottle
    [0, 128, 128],      # 6: bus
    [128, 128, 128],    # 7: car
    [64, 0, 0],         # 8: cat
    [192, 0, 0],        # 9: chair
    [64, 128, 0],       # 10: cow
    [192, 128, 0],      # 11: dining table
    [64, 0, 128],       # 12: dog
    [192, 0, 128],      # 13: horse
    [64, 128, 128],     # 14: motorbike
    [192, 128, 128],    # 15: person
    [0, 64, 0],         # 16: potted plant
    [128, 64, 0],       # 17: sheep
    [0, 192, 0],        # 18: sofa
    [128, 192, 0],      # 19: train
    [0, 64, 128],       # 20: tv/monitor
]

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
]


class VOCNormalize:
    """Custom normalization for VOC dataset"""
    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, img, target):
        # Convert PIL to tensor
        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        
        # Convert target to tensor
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        
        return img, target


def collate_fn(batch):
    """Custom collate function to handle variable-sized images"""
    images, targets = zip(*batch)
    return list(images), list(targets)


def calculate_miou(pred, target, num_classes=21):
    """
    Calculate mean Intersection over Union (mIoU)
    
    Args:
        pred: predicted segmentation mask (H, W)
        target: ground truth segmentation mask (H, W)
        num_classes: number of classes
    
    Returns:
        mIoU value and per-class IoU dict
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    ious = {}
    valid_ious = []
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            ious[cls] = None
            continue
        
        iou = intersection / union
        ious[cls] = iou
        valid_ious.append(iou)
    
    mean_iou = np.mean(valid_ious) if len(valid_ious) > 0 else 0.0
    return mean_iou, ious


def decode_segmentation_mask(mask):
    """Convert class indices to RGB color map"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_idx in range(len(VOC_COLORMAP)):
        rgb[mask == cls_idx] = VOC_COLORMAP[cls_idx]
    
    return rgb


def evaluate_model(model, dataloader, device, num_classes=21):
    """
    Evaluate model on validation set
    
    Args:
        model: model to evaluate
        dataloader: validation data loader
        device: device to run on
        num_classes: number of classes
    
    Returns:
        mean IoU, per-class IoU stats, inference time
    """
    model.eval()
    
    all_ious = []
    class_ious = {cls: [] for cls in range(num_classes)}
    inference_times = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            for img, target in zip(images, targets):
                img = img.unsqueeze(0).to(device)
                target = target.to(device)
                
                # Get original size
                h, w = target.shape
                
                # Forward pass with timing
                start_time = time.time()
                output = model(img)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Resize to original size
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
                
                # Get predictions
                pred = output.squeeze(0).argmax(0)
                
                # Calculate IoU
                miou, per_class_iou = calculate_miou(pred, target, num_classes)
                all_ious.append(miou)
                
                # Store per-class IoUs
                for cls, iou in per_class_iou.items():
                    if iou is not None:
                        class_ious[cls].append(iou)
    
    # Calculate statistics
    mean_iou = np.mean(all_ious)
    avg_inference_time = np.mean(inference_times)
    
    # Calculate per-class average IoUs
    class_avg_ious = {}
    for cls in range(num_classes):
        if len(class_ious[cls]) > 0:
            class_avg_ious[cls] = np.mean(class_ious[cls])
        else:
            class_avg_ious[cls] = None
    
    return mean_iou, class_avg_ious, avg_inference_time


def visualize_predictions(model, dataset, device, num_samples=5, save_dir='./visualizations'):
    """
    Generate visualization comparing ground truth and predictions
    
    Args:
        model: trained model
        dataset: VOC dataset
        device: device to run on
        num_samples: number of samples to visualize
        save_dir: directory to save visualizations
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    # Track results for summary
    visualization_results = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc='Generating visualizations'):
            img, target = dataset[idx]
            
            # Get original image (before normalization)
            original_img = np.array(dataset.dataset.images[idx])
            if isinstance(original_img, str):
                from PIL import Image
                original_img = np.array(Image.open(original_img).convert('RGB'))
            
            # Forward pass
            img_tensor = img.unsqueeze(0).to(device)
            h, w = target.shape
            
            output = model(img_tensor)
            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
            pred = output.squeeze(0).argmax(0).cpu()
            
            # Calculate IoU
            miou, per_class_iou = calculate_miou(pred, target)
            
            # Track for summary
            visualization_results.append({
                'index': idx,
                'filename': f'visualization_{idx}.png',
                'miou': miou,
                'quality': 'Good' if miou > 0.5 else 'Poor'
            })
            
            # Decode masks
            pred_rgb = decode_segmentation_mask(pred.numpy())
            target_rgb = decode_segmentation_mask(target.numpy())
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(original_img)
            axes[1].imshow(target_rgb, alpha=0.5)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(original_img)
            axes[2].imshow(pred_rgb, alpha=0.5)
            axes[2].set_title(f'Prediction (mIoU: {miou:.3f})')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_dir / f'visualization_{idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # Save visualization summary
    summary_path = save_dir / 'visualization_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VISUALIZATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total visualizations: {len(visualization_results)}\n\n")
        
        # Sort by mIoU (best to worst)
        visualization_results.sort(key=lambda x: x['miou'], reverse=True)
        
        f.write("RESULTS (sorted by mIoU, best to worst):\n")
        f.write("-" * 80 + "\n")
        for result in visualization_results:
            f.write(f"  {result['filename']:30s} | mIoU: {result['miou']:.4f} | {result['quality']}\n")
        
        # Separate good and poor examples
        good_examples = [r for r in visualization_results if r['quality'] == 'Good']
        poor_examples = [r for r in visualization_results if r['quality'] == 'Poor']
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CATEGORIZATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Good examples (mIoU > 0.5): {len(good_examples)}\n")
        if good_examples:
            f.write("  Best examples for report:\n")
            for result in good_examples[:3]:  # Top 3
                f.write(f"    - {result['filename']} (mIoU: {result['miou']:.4f})\n")
        
        f.write(f"\nPoor examples (mIoU <= 0.5): {len(poor_examples)}\n")
        if poor_examples:
            f.write("  Failure cases for report:\n")
            for result in poor_examples[-3:]:  # Bottom 3
                f.write(f"    - {result['filename']} (mIoU: {result['miou']:.4f})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATION FOR REPORT:\n")
        f.write("-" * 80 + "\n")
        f.write("Include in report:\n")
        f.write(f"  - 3-5 successful results from: {', '.join([r['filename'] for r in good_examples[:5]])}\n")
        f.write(f"  - 2-3 failure cases from: {', '.join([r['filename'] for r in poor_examples[-3:]])}\n")
        f.write("  - class_legend.png (color coding reference)\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nVisualizations saved to {save_dir}/")
    print(f"Summary saved to {summary_path}")


def create_legend(save_dir='./visualizations'):
    """Create a legend showing all class colors"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')
    
    # Create color patches
    for i, (color, class_name) in enumerate(zip(VOC_COLORMAP, VOC_CLASSES)):
        y_pos = 0.95 - (i * 0.045)
        
        # Color patch
        rect = Rectangle((0.1, y_pos), 0.1, 0.035, 
                        facecolor=np.array(color)/255.0, 
                        edgecolor='black', 
                        transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Class name
        ax.text(0.25, y_pos + 0.0175, class_name, 
               verticalalignment='center', 
               fontsize=10,
               transform=ax.transAxes)
    
    ax.set_title('PASCAL VOC 2012 Classes', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(save_dir / 'class_legend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Class legend saved to {save_dir / 'class_legend.png'}")


def print_results_table(results, save_path=None):
    """Print formatted results table"""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {results['model_type']}")
    print(f"KD Mode: {results['kd_mode']}")
    print(f"Parameters: {results['num_params']:,} ({results['num_params']/1e6:.2f}M)")
    print("-" * 80)
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Avg Inference Time: {results['avg_inference_time']*1000:.2f} ms")
    print(f"FPS: {1.0/results['avg_inference_time']:.2f}")
    print("-" * 80)
    print(f"FINAL SCORE: {results['score']:.4f}")
    print(f"  (Score = 4 × {results['mean_iou']:.4f} / (1 + {results['num_params']/1e6:.2f}))")
    print("=" * 80)
    
    # Per-class IoU
    print("\nPer-Class IoU:")
    print("-" * 80)
    for cls_idx, cls_name in enumerate(VOC_CLASSES):
        iou = results['class_ious'].get(cls_idx)
        if iou is not None:
            print(f"  {cls_name:20s}: {iou:.4f}")
        else:
            print(f"  {cls_name:20s}: N/A")
    print("=" * 80 + "\n")
    
    # Save to file if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Hardware information
            f.write("HARDWARE & ENVIRONMENT:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"PyTorch Version: {torch.__version__}\n")
            import platform
            f.write(f"Python Version: {platform.python_version()}\n")
            f.write(f"Operating System: {platform.system()} {platform.release()}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("MODEL INFORMATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model: {results['model_type']}\n")
            f.write(f"KD Mode: {results['kd_mode']}\n")
            f.write(f"Parameters: {results['num_params']:,} ({results['num_params']/1e6:.2f}M)\n")
            f.write(f"Checkpoint: {results['checkpoint_path']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean IoU: {results['mean_iou']:.4f}\n")
            f.write(f"Inference Time: {results['avg_inference_time']*1000:.2f} ms per image\n")
            f.write(f"Inference Speed: {results['fps']:.2f} FPS\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("FINAL SCORE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Score: {results['score']:.4f}\n")
            f.write(f"Formula: 4 × {results['mean_iou']:.4f} / (1 + {results['num_params']/1e6:.2f}M)\n")
            f.write(f"        = 4 × {results['mean_iou']:.4f} / {1 + results['num_params']/1e6:.2f}\n")
            f.write(f"        = {results['score']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("PER-CLASS IoU:\n")
            f.write("-" * 80 + "\n")
            for cls_idx, cls_name in enumerate(VOC_CLASSES):
                iou = results['class_ious'].get(cls_idx)
                if iou is not None:
                    f.write(f"  {cls_name:20s}: {iou:.4f}\n")
                else:
                    f.write(f"  {cls_name:20s}: N/A\n")
            f.write("=" * 80 + "\n")
        print(f"Results saved to {save_path}")


def main(args):
    """Main testing function"""
    
    print("=" * 80)
    print("Testing Compact Segmentation Model")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading model...")
    if args.model == 'ultracompact':
        model = UltraCompactSegmentationModel(num_classes=21, pretrained=False)
    elif args.model == 'ultracompact_v2':
        model = UltraCompactSegmentationModelV2(num_classes=21, pretrained=False)
    elif args.model == 'standard':
        model = StandardSegmentationModel(num_classes=21, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose 'ultracompact', 'ultracompact_v2', or 'standard'.")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Extract KD mode from checkpoint name
    checkpoint_name = Path(args.checkpoint).stem
    if 'response' in checkpoint_name:
        kd_mode = 'response'
    elif 'feature' in checkpoint_name:
        kd_mode = 'feature'
    else:
        kd_mode = 'none'
    
    # Prepare dataset
    print("\nPreparing dataset...")
    
    # Find dataset path (Google Colab compatible)
    # VOCSegmentation expects: root/VOCdevkit/VOC2012/
    dataset_roots = []
    
    # Check for Google Colab structure (both with and without VOCdevkit wrapper)
    colab_test_path = Path('/content/data/VOC2012_test')
    if colab_test_path.exists():
        # Check if it has VOCdevkit structure
        if (colab_test_path / 'VOCdevkit' / 'VOC2012').exists():
            dataset_roots.append(str(colab_test_path))
        # Check if files are directly in the folder (no VOCdevkit wrapper)
        elif (colab_test_path / 'JPEGImages').exists():
            # Create a symlink or use parent structure
            dataset_roots.append(str(colab_test_path.parent))
    
    # Check for local structure
    local_data_path = Path('./data')
    if local_data_path.exists():
        if (local_data_path / 'VOCdevkit' / 'VOC2012').exists():
            dataset_roots.append(str(local_data_path))
        elif (local_data_path / 'VOC2012' / 'JPEGImages').exists():
            dataset_roots.append(str(local_data_path))
    
    # Check standard paths
    dataset_roots.extend([
        '/content/data',  # Google Colab default
        './data',  # Local path
        str(Path.home() / '.cache' / 'kagglehub' / 'datasets' / 'huanghanchina' / 'pascal-voc-2012' / 'versions' / '1'),
    ])
    
    # Check dataset_path.txt
    dataset_path_file = Path('dataset_path.txt')
    if dataset_path_file.exists():
        with open(dataset_path_file, 'r') as f:
            saved_path = Path(f.read().strip())
            dataset_roots.insert(0, str(saved_path.parent.parent))
    
    dataset_root = None
    for root in dataset_roots:
        try:
            # Try to create a temporary symlink structure if needed
            root_path = Path(root)
            
            # Check if we need to create VOCdevkit/VOC2012 structure
            voc_path = root_path / 'VOC2012_test'
            if voc_path.exists() and (voc_path / 'JPEGImages').exists() and not (voc_path / 'VOCdevkit').exists():
                # Files are directly in VOC2012_test, create temporary structure
                vocdevkit_path = voc_path / 'VOCdevkit'
                voc2012_path = vocdevkit_path / 'VOC2012'
                if not voc2012_path.exists():
                    print(f"Creating VOCdevkit structure in {voc_path}")
                    vocdevkit_path.mkdir(exist_ok=True)
                    voc2012_path.symlink_to(voc_path, target_is_directory=True)
            
            test_dataset = VOCSegmentation(root=root, year='2012', image_set='val', download=False)
            dataset_root = root
            print(f"✓ Successfully loaded dataset from: {root}")
            break
        except Exception as e:
            continue
    
    if dataset_root is None:
        raise RuntimeError("Dataset not found! Please ensure dataset is in /content/data/VOC2012_test/ (Colab) or ./data/ (local)")
    
    print(f"Dataset found at: {dataset_root}")
    
    # Create dataset
    val_transform = VOCNormalize()
    val_dataset = VOCSegmentation(
        root=dataset_root,
        year='2012',
        image_set='val',
        download=False,
        transforms=val_transform
    )
    
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    mean_iou, class_ious, avg_inference_time = evaluate_model(model, val_loader, device)
    
    # Calculate score
    score = 4 * mean_iou / (1 + num_params / 1e6)
    fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    # Prepare results
    results = {
        'model_type': args.model,
        'kd_mode': kd_mode,
        'num_params': num_params,
        'mean_iou': mean_iou,
        'class_ious': class_ious,
        'avg_inference_time': avg_inference_time,
        'fps': fps,
        'score': score,
        'checkpoint_path': args.checkpoint
    }
    
    # Print results
    save_path = Path(args.save_dir) / f'results_{args.model}_{kd_mode}.txt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print_results_table(results, save_path)
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        vis_dir = Path(args.save_dir) / f'visualizations_{args.model}_{kd_mode}'
        visualize_predictions(model, val_dataset, device, num_samples=args.num_vis, save_dir=vis_dir)
        create_legend(vis_dir)
    
    # Save results as pickle
    import pickle
    results_pkl_path = Path(args.save_dir) / f'results_{args.model}_{kd_mode}.pkl'
    with open(results_pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_pkl_path}")
    
    print("\n" + "=" * 80)
    print("Testing completed!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test compact segmentation model')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, choices=['ultracompact', 'ultracompact_v2', 'standard'],
                        help='Model type to test')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Evaluation arguments
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization images')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='Number of visualizations to generate')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    main(args)
