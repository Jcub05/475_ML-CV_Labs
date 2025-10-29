import os
import sys
import time
import argparse
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import models
sys.path.append('SnoutNet')
sys.path.append('SnoutNet-A')
sys.path.append('SnoutNet-V')
from model import SnoutNet
from model_alexnet import SnoutNetAlexNet
from model_vgg16 import SnoutNetVGG16

from datamodule import get_dataloaders


def calculate_euclidean_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate Euclidean distance between predicted and target coordinates."""
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=1))


def get_model(model_type: str, pretrained: bool = True):
    """
    Get model based on type.
    
    Args:
        model_type: 'snoutnet', 'alexnet', or 'vgg16'
        pretrained: If True, use pretrained weights (for alexnet/vgg16)
    
    Returns:
        Model instance
    """
    if model_type == 'snoutnet':
        return SnoutNet()
    elif model_type == 'alexnet':
        return SnoutNetAlexNet(pretrained=pretrained, freeze_backbone=False)
    elif model_type == 'vgg16':
        return SnoutNetVGG16(pretrained=pretrained, freeze_backbone=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_distance = 0.0
    num_samples = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        batch_size = images.size(0)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics - weight loss by batch size for correct averaging
        total_loss += loss.item() * batch_size
        distances = calculate_euclidean_distance(predictions, targets)
        total_distance += distances.sum().item()
        num_samples += batch_size
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, '
                  f'Avg Distance: {distances.mean().item():.2f}px')
    
    avg_loss = total_loss / num_samples
    avg_distance = total_distance / num_samples
    
    return {'loss': avg_loss, 'distance': avg_distance}


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                  device: torch.device) -> Dict[str, float]:
    """Validate the model for one epoch."""
    model.eval()
    total_loss = 0.0
    total_distance = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            batch_size = images.size(0)
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Statistics - weight loss by batch size for correct averaging
            total_loss += loss.item() * batch_size
            distances = calculate_euclidean_distance(predictions, targets)
            total_distance += distances.sum().item()
            num_samples += batch_size
    
    avg_loss = total_loss / num_samples
    avg_distance = total_distance / num_samples
    
    return {'loss': avg_loss, 'distance': avg_distance}


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        train_distances: List[float], val_distances: List[float],
                        save_path: str):
    """Plot and save training curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distance plot
    ax2.plot(epochs, train_distances, 'b-', label='Training Distance', linewidth=2)
    ax2.plot(epochs, val_distances, 'r-', label='Validation Distance', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Distance (pixels)')
    ax2.set_title('Training and Validation Localization Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}")


def train_model(args):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        normalize_imagenet=args.normalize_imagenet
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"Creating {args.model_type.upper()} model...")
    model = get_model(args.model_type, pretrained=True).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training history
    train_losses = []
    val_losses = []
    train_distances = []
    val_distances = []
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, f'best_{args.model_type}.pth')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Model: {args.model_type.upper()}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Loss function: {criterion.__class__.__name__}")
    print("-" * 50)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Store metrics
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_distances.append(train_metrics['distance'])
        val_distances.append(val_metrics['distance'])
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Distance: {train_metrics['distance']:.2f}px")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Distance: {val_metrics['distance']:.2f}px")
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_distance': train_metrics['distance'],
                'val_distance': val_metrics['distance'],
                'model_type': args.model_type,
            }, best_model_path)
            print(f"New best model saved! (Val Loss: {best_val_loss:.4f})")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    # Plot training curves
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, train_distances, val_distances, plot_path)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f'final_{args.model_type}.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'train_distance': train_distances[-1],
        'val_distance': val_distances[-1],
        'model_type': args.model_type,
    }, final_model_path)
    
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training curves saved to: {plot_path}")
    
    return model, train_losses, val_losses, train_distances, val_distances


def main():
    parser = argparse.ArgumentParser(description='Train models for pet nose localization')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, required=True, choices=['snoutnet', 'alexnet', 'vgg16'],
                       help='Model type: snoutnet, alexnet, or vgg16')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='.', 
                       help='Root directory containing oxford-iiit-pet-noses dataset')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of workers for data loading')
    parser.add_argument('--normalize_imagenet', action='store_true',
                       help='Use ImageNet normalization')
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation (horizontal flip + color jitter)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay for regularization')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./training_output', 
                       help='Directory to save models and plots')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model type: {args.model_type.upper()}")
    print(f"Data root: {args.data_root}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Output directory: {args.output_dir}")
    print(f"ImageNet normalization: {args.normalize_imagenet}")
    print(f"Data augmentation: {args.augment}")
    print("=" * 60)
    
    # Train the model
    model, train_losses, val_losses, train_distances, val_distances = train_model(args)
    
    print("\nTraining Summary:")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Final training distance: {train_distances[-1]:.2f}px")
    print(f"Final validation distance: {val_distances[-1]:.2f}px")
    print(f"Best validation loss: {min(val_losses):.4f}")


if __name__ == '__main__':
    main()

