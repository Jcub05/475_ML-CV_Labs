import os
import sys
import argparse
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import models
sys.path.append('SnoutNet')
sys.path.append('SnoutNet-A')
sys.path.append('SnoutNet-V')
sys.path.append('SnoutNet-Ensemble')
from model import SnoutNet
from model_alexnet import SnoutNetAlexNet
from model_vgg16 import SnoutNetVGG16
from model_ensemble import SnoutNetEnsemble

from datamodule import get_datasets


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


def denormalize(tensor: torch.Tensor, normalize_imagenet: bool = False) -> np.ndarray:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Image tensor [C, H, W] in [0,1] or ImageNet normalized
        normalize_imagenet: If True, reverse ImageNet normalization
    
    Returns:
        numpy array [H, W, C] in [0, 255]
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    
    if normalize_imagenet:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
    
    # Clip to [0, 1] and convert to [0, 255]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img


def calculate_euclidean_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Euclidean distance between predicted and target coordinates."""
    return torch.sqrt(torch.sum((pred - target) ** 2)).item()


def visualize_predictions(model: nn.Module, dataset, device: torch.device, 
                         num_samples: int, normalize_imagenet: bool, 
                         save_path: str, model_type: str):
    """
    Visualize model predictions on test samples.
    
    Args:
        model: Trained model
        dataset: Test dataset
        device: Device to run inference on
        num_samples: Number of samples to visualize
        normalize_imagenet: Whether ImageNet normalization was used
        save_path: Path to save visualization
        model_type: Type of model (for title)
    """
    model.eval()
    
    # Select random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Create grid of subplots
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    print(f"Visualizing {num_samples} samples...")
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            image_tensor, target = dataset[idx]
            
            # Denormalize for visualization
            image_np = denormalize(image_tensor, normalize_imagenet)
            
            # Get prediction
            image_batch = image_tensor.unsqueeze(0).to(device)
            prediction = model(image_batch).squeeze(0).cpu()
            
            # Calculate error
            distance = calculate_euclidean_distance(prediction, target)
            
            # Plot
            ax = axes[i]
            ax.imshow(image_np)
            
            # Plot ground truth (green)
            ax.plot(target[0].item(), target[1].item(), 'go', markersize=15, 
                   markeredgewidth=2, markeredgecolor='white', label='Ground Truth')
            
            # Plot prediction (red)
            ax.plot(prediction[0].item(), prediction[1].item(), 'rx', markersize=15, 
                   markeredgewidth=3, label='Prediction')
            
            # Draw line between prediction and ground truth
            ax.plot([target[0].item(), prediction[0].item()],
                   [target[1].item(), prediction[1].item()],
                   'r--', linewidth=2, alpha=0.7)
            
            ax.set_title(f'Sample {idx}\nError: {distance:.2f}px', fontsize=12)
            ax.axis('off')
            ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{model_type.upper()} - Nose Localization Predictions', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_path}")


def evaluate_all_distances(model: nn.Module, dataset, device: torch.device) -> List[Tuple[int, float]]:
    """Run inference across the dataset and collect (index, euclidean_distance)."""
    model.eval()
    results: List[Tuple[int, float]] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            image_tensor, target = dataset[idx]
            pred = model(image_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
            dist = calculate_euclidean_distance(pred, target)
            results.append((idx, dist))
    return results


def visualize_predictions_indices(model: nn.Module, dataset, device: torch.device,
                                  indices: List[int], normalize_imagenet: bool,
                                  save_path: str, model_type: str, title_suffix: str = ""):
    """Visualize predictions for a fixed list of dataset indices."""
    model.eval()
    num_samples = len(indices)
    cols = min(4, max(1, num_samples))
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image_tensor, target = dataset[idx]
            image_np = denormalize(image_tensor, normalize_imagenet)
            prediction = model(image_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
            distance = calculate_euclidean_distance(prediction, target)

            ax = axes[i]
            ax.imshow(image_np)
            ax.plot(target[0].item(), target[1].item(), 'go', markersize=15,
                    markeredgewidth=2, markeredgecolor='white', label='Ground Truth')
            ax.plot(prediction[0].item(), prediction[1].item(), 'rx', markersize=15,
                    markeredgewidth=3, label='Prediction')
            ax.plot([target[0].item(), prediction[0].item()],
                    [target[1].item(), prediction[1].item()],
                    'r--', linewidth=2, alpha=0.7)
            ax.set_title(f'idx {idx} | error {distance:.2f}px', fontsize=11)
            ax.axis('off')
            ax.legend(loc='upper right', fontsize=8)

    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    suptitle = f'{model_type.upper()} - Predictions'
    if title_suffix:
        suptitle += f' ({title_suffix})'
    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def visualize_best_worst_combined(model: nn.Module, dataset, device: torch.device,
                                  best_indices: List[int], worst_indices: List[int],
                                  normalize_imagenet: bool, save_path: str,
                                  model_type: str):
    """Create one figure: top row=4 best, bottom row=4 worst."""
    model.eval()
    best_indices = list(best_indices)[:4]
    worst_indices = list(worst_indices)[:4]

    cols = 4
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 10))

    with torch.no_grad():
        # Best row
        for i in range(cols):
            ax = axes[0, i]
            if i < len(best_indices):
                idx = best_indices[i]
                image_tensor, target = dataset[idx]
                image_np = denormalize(image_tensor, normalize_imagenet)
                prediction = model(image_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
                distance = calculate_euclidean_distance(prediction, target)
                ax.imshow(image_np)
                ax.plot(target[0].item(), target[1].item(), 'go', markersize=15,
                        markeredgewidth=2, markeredgecolor='white', label='GT')
                ax.plot(prediction[0].item(), prediction[1].item(), 'rx', markersize=15,
                        markeredgewidth=3, label='Pred')
                ax.plot([target[0].item(), prediction[0].item()],
                        [target[1].item(), prediction[1].item()], 'r--', linewidth=2, alpha=0.7)
                ax.set_title(f'Best {i+1} | idx {idx} | {distance:.2f}px', fontsize=11)
            ax.axis('off')

        # Worst row
        for j in range(cols):
            ax = axes[1, j]
            if j < len(worst_indices):
                idx = worst_indices[j]
                image_tensor, target = dataset[idx]
                image_np = denormalize(image_tensor, normalize_imagenet)
                prediction = model(image_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
                distance = calculate_euclidean_distance(prediction, target)
                ax.imshow(image_np)
                ax.plot(target[0].item(), target[1].item(), 'go', markersize=15,
                        markeredgewidth=2, markeredgecolor='white', label='GT')
                ax.plot(prediction[0].item(), prediction[1].item(), 'rx', markersize=15,
                        markeredgewidth=3, label='Pred')
                ax.plot([target[0].item(), prediction[0].item()],
                        [target[1].item(), prediction[1].item()], 'r--', linewidth=2, alpha=0.7)
                ax.set_title(f'Worst {j+1} | idx {idx} | {distance:.2f}px', fontsize=11)
            ax.axis('off')

    plt.suptitle(f'{model_type.upper()} - 4 Best (Top) | 4 Worst (Bottom)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions for pet nose localization')
    
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
    parser.add_argument('--compare_models_combined', action='store_true',
                       help='Create a single combined figure comparing SnoutNet, AlexNet, VGG16, and Ensemble (4 best top, 4 worst bottom per model)')
    
    parser.add_argument('--data_root', type=str, default='.',
                       help='Root directory containing oxford-iiit-pet-noses dataset')
    parser.add_argument('--normalize_imagenet', action='store_true',
                       help='Use ImageNet normalization (not used for ensemble)')
    parser.add_argument('--num_samples', type=int, default=8,
                       help='Number of samples to visualize (when --select=random)')
    parser.add_argument('--select', type=str, choices=['random', 'best4', 'worst4', 'both', 'combined'], default='random',
                       help='Selection: random N, 4 best, 4 worst, both (two files), or combined (single 2x4 panel)')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Special mode: combined comparison of all models
    if args.compare_models_combined:
        # Validate required paths
        if not (args.snoutnet_path and args.alexnet_path and args.vgg16_path):
            raise ValueError("--compare_models_combined requires --snoutnet_path, --alexnet_path, and --vgg16_path (these are also used to build the ensemble)")

        print("Building models for combined comparison: SnoutNet, AlexNet, VGG16, and Ensemble...")

        # Build models
        # 1) Single models
        # SnoutNet
        print(f"Loading SnoutNet from {args.snoutnet_path}")
        ck_sn = torch.load(args.snoutnet_path, map_location=device)
        sn_model = get_model('snoutnet').to(device)
        sn_model.load_state_dict(ck_sn['model_state_dict'])

        # AlexNet
        print(f"Loading AlexNet from {args.alexnet_path}")
        ck_ax = torch.load(args.alexnet_path, map_location=device)
        ax_model = get_model('alexnet').to(device)
        ax_model.load_state_dict(ck_ax['model_state_dict'])

        # VGG16
        print(f"Loading VGG16 from {args.vgg16_path}")
        ck_vg = torch.load(args.vgg16_path, map_location=device)
        vg_model = get_model('vgg16').to(device)
        vg_model.load_state_dict(ck_vg['model_state_dict'])

        # 2) Ensemble (uses same three paths)
        ens_model = get_model('ensemble', device=device, ensemble_paths={
            'snoutnet': args.snoutnet_path,
            'alexnet': args.alexnet_path,
            'vgg16': args.vgg16_path,
        })

        # Datasets (normalization differs by model)
        print("Loading test datasets for plain and ImageNet-normalized inputs...")
        _, test_plain = get_datasets(root=args.data_root, augment=False, normalize_imagenet=False)
        _, test_imn = get_datasets(root=args.data_root, augment=False, normalize_imagenet=True)

        # Evaluate distances and pick indices
        print("Evaluating distances for each model to find top-4 and bottom-4...")
        sn_scores = evaluate_all_distances(sn_model, test_plain, device)
        ax_scores = evaluate_all_distances(ax_model, test_imn, device)
        vg_scores = evaluate_all_distances(vg_model, test_imn, device)
        # Ensemble expects plain-normalized dataset
        ens_scores = evaluate_all_distances(ens_model, test_plain, device)

        def top_bottom_indices(scores):
            asc = sorted(scores, key=lambda x: x[1])
            desc = sorted(scores, key=lambda x: x[1], reverse=True)
            return [i for i, _ in asc[:4]], [i for i, _ in desc[:4]]

        sn_best, sn_worst = top_bottom_indices(sn_scores)
        ax_best, ax_worst = top_bottom_indices(ax_scores)
        vg_best, vg_worst = top_bottom_indices(vg_scores)
        ens_best, ens_worst = top_bottom_indices(ens_scores)

        # Compose a large figure: 8 rows x 4 cols (two rows per model)
        print("Composing combined 2x4 panels per model into one figure...")
        rows, cols = 8, 4
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.5 * rows))

        def render_row_pair(row_start, model_name, model, dataset, best_idxs, worst_idxs, normalize_imagenet_flag):
            with torch.no_grad():
                # Best row
                r = row_start
                for i in range(4):
                    axp = axes[r, i]
                    if i < len(best_idxs):
                        idx = best_idxs[i]
                        img_t, tgt = dataset[idx]
                        img_np = denormalize(img_t, normalize_imagenet_flag)
                        pred = model(img_t.unsqueeze(0).to(device)).squeeze(0).cpu()
                        dist = calculate_euclidean_distance(pred, tgt)
                        axp.imshow(img_np)
                        axp.plot(tgt[0].item(), tgt[1].item(), 'go', markersize=10, markeredgewidth=2, markeredgecolor='white')
                        axp.plot(pred[0].item(), pred[1].item(), 'rx', markersize=10, markeredgewidth=3)
                        axp.plot([tgt[0].item(), pred[0].item()], [tgt[1].item(), pred[1].item()], 'r--', linewidth=2, alpha=0.7)
                        axp.set_title(f'{model_name} Best {i+1}\nidx {idx} | {dist:.2f}px', fontsize=9)
                    axp.axis('off')
                # Worst row
                r = row_start + 1
                for j in range(4):
                    axp = axes[r, j]
                    if j < len(worst_idxs):
                        idx = worst_idxs[j]
                        img_t, tgt = dataset[idx]
                        img_np = denormalize(img_t, normalize_imagenet_flag)
                        pred = model(img_t.unsqueeze(0).to(device)).squeeze(0).cpu()
                        dist = calculate_euclidean_distance(pred, tgt)
                        axp.imshow(img_np)
                        axp.plot(tgt[0].item(), tgt[1].item(), 'go', markersize=10, markeredgewidth=2, markeredgecolor='white')
                        axp.plot(pred[0].item(), pred[1].item(), 'rx', markersize=10, markeredgewidth=3)
                        axp.plot([tgt[0].item(), pred[0].item()], [tgt[1].item(), pred[1].item()], 'r--', linewidth=2, alpha=0.7)
                        axp.set_title(f'{model_name} Worst {j+1}\nidx {idx} | {dist:.2f}px', fontsize=9)
                    axp.axis('off')

        render_row_pair(0, 'SnoutNet', sn_model, test_plain, sn_best, sn_worst, False)
        render_row_pair(2, 'AlexNet', ax_model, test_imn, ax_best, ax_worst, True)
        render_row_pair(4, 'VGG16', vg_model, test_imn, vg_best, vg_worst, True)
        render_row_pair(6, 'Ensemble', ens_model, test_plain, ens_best, ens_worst, False)

        plt.suptitle('4 Best (Top) and 4 Worst (Bottom) per Model: SnoutNet | AlexNet | VGG16 | Ensemble', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        combined_path = os.path.join('.', 'combined_best4_worst4_all_models.png')
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            combined_path = os.path.join(args.output_dir, 'combined_best4_worst4_all_models.png')
        except Exception:
            pass
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined comparison saved to {combined_path}")
        print("\nVisualization complete!")
        return

    # Determine if ensemble or single model
    if args.model_type == 'ensemble' or (args.snoutnet_path and args.alexnet_path and args.vgg16_path):
        # Ensemble mode
        model_type = 'ensemble'
        print("Visualizing ENSEMBLE model...")
        
        if not all([args.snoutnet_path, args.alexnet_path, args.vgg16_path]):
            raise ValueError("All three model paths required for ensemble: --snoutnet_path, --alexnet_path, --vgg16_path")
        
        ensemble_paths = {
            'snoutnet': args.snoutnet_path,
            'alexnet': args.alexnet_path,
            'vgg16': args.vgg16_path
        }
        
        model = get_model('ensemble', device=device, ensemble_paths=ensemble_paths)
        
    else:
        # Single model mode
        if args.model_path is None:
            raise ValueError("--model_path required for single model visualization")
        
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
    
    # Load test dataset
    print("Loading test dataset...")
    # Ensemble doesn't need ImageNet normalization (handles internally)
    use_imagenet_norm = args.normalize_imagenet if model_type != 'ensemble' else False
    _, test_dataset = get_datasets(
        root=args.data_root,
        augment=False,
        normalize_imagenet=use_imagenet_norm
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Route by selection mode
    if args.select == 'random':
        save_path = os.path.join(args.output_dir, f'{model_type}_predictions.png')
        visualize_predictions(
            model=model,
            dataset=test_dataset,
            device=device,
            num_samples=args.num_samples,
            normalize_imagenet=use_imagenet_norm,
            save_path=save_path,
            model_type=model_type
        )
    else:
        print("Evaluating full test set to rank predictions by error...")
        scores = evaluate_all_distances(model, test_dataset, device)
        scores_sorted_asc = sorted(scores, key=lambda x: x[1])
        scores_sorted_desc = sorted(scores, key=lambda x: x[1], reverse=True)

        if args.select in ('best4', 'both'):
            best_indices = [idx for idx, _ in scores_sorted_asc[:4]]
            save_best = os.path.join(args.output_dir, f'{model_type}_best4.png')
            visualize_predictions_indices(
                model=model,
                dataset=test_dataset,
                device=device,
                indices=best_indices,
                normalize_imagenet=use_imagenet_norm,
                save_path=save_best,
                model_type=model_type,
                title_suffix='4 Best'
            )

        if args.select in ('worst4', 'both'):
            worst_indices = [idx for idx, _ in scores_sorted_desc[:4]]
            save_worst = os.path.join(args.output_dir, f'{model_type}_worst4.png')
            visualize_predictions_indices(
                model=model,
                dataset=test_dataset,
                device=device,
                indices=worst_indices,
                normalize_imagenet=use_imagenet_norm,
                save_path=save_worst,
                model_type=model_type,
                title_suffix='4 Worst'
            )

        if args.select == 'combined':
            best_indices = [idx for idx, _ in scores_sorted_asc[:4]]
            worst_indices = [idx for idx, _ in scores_sorted_desc[:4]]
            save_combined = os.path.join(args.output_dir, f'{model_type}_best4_worst4.png')
            visualize_best_worst_combined(
                model=model,
                dataset=test_dataset,
                device=device,
                best_indices=best_indices,
                worst_indices=worst_indices,
                normalize_imagenet=use_imagenet_norm,
                save_path=save_combined,
                model_type=model_type,
            )
    
    print(f"\nVisualization complete!")


if __name__ == '__main__':
    main()

