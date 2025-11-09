import os
import sys
import argparse
import time
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Import models
sys.path.append('SnoutNet')
sys.path.append('SnoutNet-A')
sys.path.append('SnoutNet-V')
sys.path.append('SnoutNet-Ensemble')
from model import SnoutNet
from model_alexnet import SnoutNetAlexNet
from model_vgg16 import SnoutNetVGG16
from model_ensemble import SnoutNetEnsemble

from datamodule import get_dataloaders


def get_model(model_type: str, device: torch.device = None, ensemble_paths: dict = None):
    """
    Get model based on type.
    
    Args:
        model_type: 'snoutnet', 'alexnet', 'vgg16', or 'ensemble'
        device: Device for ensemble model
        ensemble_paths: Dict with 'snoutnet', 'alexnet', 'vgg16' paths (for ensemble)
    
    Returns:
        Model instance
    """
    if model_type == 'snoutnet':
        return SnoutNet()
    elif model_type == 'alexnet':
        return SnoutNetAlexNet(pretrained=False, freeze_backbone=False)
    elif model_type == 'vgg16':
        return SnoutNetVGG16(pretrained=False, freeze_backbone=False)
    elif model_type == 'ensemble':
        if ensemble_paths is None:
            raise ValueError("ensemble_paths required for ensemble model")
        return SnoutNetEnsemble(
            snoutnet_path=ensemble_paths['snoutnet'],
            alexnet_path=ensemble_paths['alexnet'],
            vgg16_path=ensemble_paths['vgg16'],
            device=str(device)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def calculate_euclidean_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate Euclidean distance between predicted and target coordinates."""
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=1))


def test_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Test the model and calculate localization accuracy statistics.
    
    Returns:
        Dictionary containing min, mean, max, std of Euclidean distances
    """
    model.eval()
    all_distances: List[float] = []
    total_images = 0
    
    print("Testing model...")
    # Start wall-clock timer; include data transfer + inference
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)
            bsz = images.size(0)
            
            # Forward pass
            predictions = model(images)
            
            # Calculate distances
            distances = calculate_euclidean_distance(predictions, targets)
            all_distances.extend(distances.cpu().numpy().tolist())
            total_images += bsz
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
    # Stop timer; sync if CUDA for accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_time
    
    # Calculate statistics
    all_distances_np = np.array(all_distances)
    
    # Overall statistics
    results = {
        'min': float(np.min(all_distances_np)),
        'mean': float(np.mean(all_distances_np)),
        'max': float(np.max(all_distances_np)),
        'std': float(np.std(all_distances_np)),
        'median': float(np.median(all_distances_np)),
        'num_samples': len(all_distances),
        # Timing metrics
        'elapsed_s': float(elapsed_s),
        'ms_per_image': float((elapsed_s * 1000.0) / max(1, total_images)),
        'images_per_s': float(total_images / max(1e-9, elapsed_s)),
    }
    
    # Calculate statistics for 4 best and 4 worst samples
    sorted_distances = np.sort(all_distances_np)
    best_4 = sorted_distances[:4]
    worst_4 = sorted_distances[-4:]
    
    results['best_4'] = {
        'min': float(np.min(best_4)),
        'mean': float(np.mean(best_4)),
        'max': float(np.max(best_4)),
        'std': float(np.std(best_4)),
    }
    
    results['worst_4'] = {
        'min': float(np.min(worst_4)),
        'mean': float(np.mean(worst_4)),
        'max': float(np.max(worst_4)),
        'std': float(np.std(worst_4)),
    }
    
    return results, all_distances


def plot_error_distribution(distances: List[float], save_path: str):
    """Plot and save error distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(distances, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.2f}px')
    ax1.axvline(np.median(distances), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(distances):.2f}px')
    ax1.set_xlabel('Euclidean Distance (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Localization Errors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_distances = np.sort(distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    ax2.plot(sorted_distances, cumulative, linewidth=2, color='blue')
    ax2.set_xlabel('Euclidean Distance (pixels)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution of Localization Errors')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(0.95, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error distribution plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test trained model for pet nose localization')
    
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model checkpoint (not used for ensemble)')
    parser.add_argument('--model_type', type=str, choices=['snoutnet', 'alexnet', 'vgg16', 'ensemble'],
                       help='Model type (auto-detected from checkpoint if not provided)')
    
    # Ensemble-specific arguments
    parser.add_argument('--snoutnet_path', type=str,
                       help='Path to SnoutNet checkpoint (for ensemble)')
    parser.add_argument('--alexnet_path', type=str,
                       help='Path to AlexNet checkpoint (for ensemble)')
    parser.add_argument('--vgg16_path', type=str,
                       help='Path to VGG16 checkpoint (for ensemble)')
    
    parser.add_argument('--data_root', type=str, default='.',
                       help='Root directory containing oxford-iiit-pet-noses dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--normalize_imagenet', action='store_true',
                       help='Use ImageNet normalization (not used for ensemble)')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Directory to save test results')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine if ensemble or single model
    if args.model_type == 'ensemble' or (args.snoutnet_path and args.alexnet_path and args.vgg16_path):
        # Ensemble mode
        model_type = 'ensemble'
        print("Testing ENSEMBLE model...")
        
        if not all([args.snoutnet_path, args.alexnet_path, args.vgg16_path]):
            raise ValueError("All three model paths required for ensemble: --snoutnet_path, --alexnet_path, --vgg16_path")
        
        ensemble_paths = {
            'snoutnet': args.snoutnet_path,
            'alexnet': args.alexnet_path,
            'vgg16': args.vgg16_path
        }
        
        model = get_model('ensemble', device=device, ensemble_paths=ensemble_paths)
        # No need to load state dict for ensemble - already loaded in constructor
        
    else:
        # Single model mode
        if args.model_path is None:
            raise ValueError("--model_path required for single model testing")
        
        # Load checkpoint
        print(f"Loading model from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Determine model type
        if args.model_type is None:
            if 'model_type' in checkpoint:
                model_type = checkpoint['model_type']
                print(f"Model type auto-detected: {model_type}")
            else:
                raise ValueError("Model type not found in checkpoint. Please specify --model_type")
        else:
            model_type = args.model_type
        
        # Create model
        print(f"Creating {model_type.upper()} model...")
        model = get_model(model_type).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_loss' in checkpoint:
            print(f"Validation loss at save: {checkpoint['val_loss']:.4f}")
        if 'val_distance' in checkpoint:
            print(f"Validation distance at save: {checkpoint['val_distance']:.2f}px")
    
    # Load test dataset
    print("Loading test dataset...")
    # Ensemble doesn't need ImageNet normalization (handles internally)
    use_imagenet_norm = args.normalize_imagenet if model_type != 'ensemble' else False
    _, test_loader = get_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,  # No augmentation for testing
        normalize_imagenet=use_imagenet_norm
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test the model
    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)
    
    results, all_distances = test_model(model, test_loader, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("LOCALIZATION ACCURACY STATISTICS")
    print("=" * 60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Total inference time: {results['elapsed_s']:.3f} s")
    print(f"Throughput: {results['images_per_s']:.2f} images/s  ({results['ms_per_image']:.2f} ms/img)")
    print(f"Minimum error: {results['min']:.2f} pixels")
    print(f"Mean error: {results['mean']:.2f} pixels")
    print(f"Median error: {results['median']:.2f} pixels")
    print(f"Maximum error: {results['max']:.2f} pixels")
    print(f"Standard deviation: {results['std']:.2f} pixels")
    print("=" * 60)
    print("\n4 BEST SAMPLES STATISTICS")
    print("=" * 60)
    print(f"Minimum error: {results['best_4']['min']:.2f} pixels")
    print(f"Mean error: {results['best_4']['mean']:.2f} pixels")
    print(f"Maximum error: {results['best_4']['max']:.2f} pixels")
    print(f"Standard deviation: {results['best_4']['std']:.2f} pixels")
    print("=" * 60)
    print("\n4 WORST SAMPLES STATISTICS")
    print("=" * 60)
    print(f"Minimum error: {results['worst_4']['min']:.2f} pixels")
    print(f"Mean error: {results['worst_4']['mean']:.2f} pixels")
    print(f"Maximum error: {results['worst_4']['max']:.2f} pixels")
    print(f"Standard deviation: {results['worst_4']['std']:.2f} pixels")
    print("=" * 60)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, f'{model_type}_test_results.txt')
    with open(results_file, 'w') as f:
        f.write("LOCALIZATION ACCURACY STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_type.upper()}\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Number of samples: {results['num_samples']}\n")
        f.write(f"Total inference time (s): {results['elapsed_s']:.6f}\n")
        f.write(f"Throughput (images/s): {results['images_per_s']:.3f}\n")
        f.write(f"Latency (ms/img): {results['ms_per_image']:.3f}\n")
        f.write(f"Minimum error: {results['min']:.2f} pixels\n")
        f.write(f"Mean error: {results['mean']:.2f} pixels\n")
        f.write(f"Median error: {results['median']:.2f} pixels\n")
        f.write(f"Maximum error: {results['max']:.2f} pixels\n")
        f.write(f"Standard deviation: {results['std']:.2f} pixels\n")
        f.write("=" * 60 + "\n")
        f.write("\n4 BEST SAMPLES STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Minimum error: {results['best_4']['min']:.2f} pixels\n")
        f.write(f"Mean error: {results['best_4']['mean']:.2f} pixels\n")
        f.write(f"Maximum error: {results['best_4']['max']:.2f} pixels\n")
        f.write(f"Standard deviation: {results['best_4']['std']:.2f} pixels\n")
        f.write("=" * 60 + "\n")
        f.write("\n4 WORST SAMPLES STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Minimum error: {results['worst_4']['min']:.2f} pixels\n")
        f.write(f"Mean error: {results['worst_4']['mean']:.2f} pixels\n")
        f.write(f"Maximum error: {results['worst_4']['max']:.2f} pixels\n")
        f.write(f"Standard deviation: {results['worst_4']['std']:.2f} pixels\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Plot error distribution
    plot_path = os.path.join(args.output_dir, f'{model_type}_error_distribution.png')
    plot_error_distribution(all_distances, plot_path)
    
    print(f"\nTesting complete!")


if __name__ == '__main__':
    main()

