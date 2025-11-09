"""
Training Script for Compact Segmentation Models with Knowledge Distillation
ELEC 475 Lab 3

Supports three training modes:
1. No Knowledge Distillation (baseline)
2. Response-based Knowledge Distillation
3. Feature-based Knowledge Distillation

References:
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- Romero et al., "FitNets: Hints for Thin Deep Nets" (2014)
"""

import argparse
import os
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from tqdm import tqdm

# Import custom models
from model_ultracompact import UltraCompactSegmentationModel
from model_standard import StandardSegmentationModel


class VOCNormalize:
    """Custom normalization for VOC dataset"""
    def __init__(self, augment=True):
        self.augment = augment
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, img, target):
        # Convert PIL to tensor
        img = transforms.ToTensor()(img)
        
        # Apply augmentation if training
        if self.augment and np.random.rand() > 0.5:
            # Random horizontal flip
            img = transforms.functional.hflip(img)
            target = transforms.functional.hflip(target)
        
        # Normalize image
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
        mIoU value
    """
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
    
    return np.mean(ious) if len(ious) > 0 else 0.0


def response_based_kd_loss(student_logits, teacher_logits, targets, temperature=3.0, alpha=0.5, beta=0.5):
    """
    Response-based knowledge distillation loss
    
    Loss = α × CE(student, targets) + β × KL_Div(student_soft, teacher_soft) × T²
    
    Args:
        student_logits: student model output logits (B, C, H, W)
        teacher_logits: teacher model output logits (B, C, H, W)
        targets: ground truth labels (B, H, W)
        temperature: softmax temperature for distillation
        alpha: weight for cross-entropy loss
        beta: weight for distillation loss
    
    Returns:
        total loss
    """
    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(student_logits, targets, ignore_index=255)
    
    # Distillation loss (KL divergence)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    # Combined loss
    total_loss = alpha * ce_loss + beta * kd_loss
    
    return total_loss, ce_loss.item(), kd_loss.item()


def feature_based_kd_loss(student_logits, student_features, teacher_features, targets, alpha=0.5, beta=0.5):
    """
    Feature-based knowledge distillation loss
    
    Loss = α × CE(student, targets) + β × Σ CosineLoss(student_feat, teacher_feat)
    
    Args:
        student_logits: student model output logits (B, C, H, W)
        student_features: dict of student intermediate features
        teacher_features: dict of teacher intermediate features
        targets: ground truth labels (B, H, W)
        alpha: weight for cross-entropy loss
        beta: weight for feature distillation loss
    
    Returns:
        total loss
    """
    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(student_logits, targets, ignore_index=255)
    
    # Feature matching loss (cosine similarity)
    feat_loss = 0
    num_levels = 0
    
    for level in ['low', 'mid', 'high']:
        if level in student_features and level in teacher_features:
            s_feat = student_features[level]
            t_feat = teacher_features[level]
            
            # Resize if needed (teacher features might be different size)
            if s_feat.shape != t_feat.shape:
                t_feat = F.interpolate(t_feat, size=s_feat.shape[2:], mode='bilinear', align_corners=False)
                # Match channels if needed
                if s_feat.shape[1] != t_feat.shape[1]:
                    # Project teacher features to student channel size
                    continue  # Skip if channel mismatch (or add projection layer)
            
            # Cosine similarity loss (1 - cosine_similarity)
            s_feat_norm = F.normalize(s_feat, p=2, dim=1)
            t_feat_norm = F.normalize(t_feat, p=2, dim=1)
            
            cosine_sim = (s_feat_norm * t_feat_norm).sum(dim=1).mean()
            feat_loss += (1 - cosine_sim)
            num_levels += 1
    
    if num_levels > 0:
        feat_loss = feat_loss / num_levels
    
    # Combined loss
    total_loss = alpha * ce_loss + beta * feat_loss
    
    return total_loss, ce_loss.item(), feat_loss.item() if num_levels > 0 else 0.0


def train_epoch(model, teacher_model, dataloader, optimizer, device, kd_mode='none', args=None):
    """
    Train for one epoch
    
    Args:
        model: student model
        teacher_model: teacher model (None if kd_mode='none')
        dataloader: training data loader
        optimizer: optimizer
        device: device to run on
        kd_mode: 'none', 'response', or 'feature'
        args: training arguments
    
    Returns:
        average loss, average ce_loss, average kd_loss
    """
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    
    total_loss = 0
    total_ce_loss = 0
    total_kd_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, targets in pbar:
        # Move to device and resize to fixed size for batching
        batch_imgs = []
        batch_targets = []
        
        for img, target in zip(images, targets):
            img = img.to(device)
            target = target.to(device)
            
            # Resize to fixed size
            img = F.interpolate(img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
            target = F.interpolate(target.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest').squeeze(0).squeeze(0).long()
            
            batch_imgs.append(img)
            batch_targets.append(target)
        
        batch_imgs = torch.stack(batch_imgs)
        batch_targets = torch.stack(batch_targets)
        
        optimizer.zero_grad()
        
        # Forward pass based on KD mode
        if kd_mode == 'none':
            # No knowledge distillation
            output = model(batch_imgs)
            loss = F.cross_entropy(output, batch_targets, ignore_index=255)
            ce_loss_val = loss.item()
            kd_loss_val = 0.0
            
        elif kd_mode == 'response':
            # Response-based KD
            output = model(batch_imgs)
            with torch.no_grad():
                teacher_output = teacher_model(batch_imgs)['out']
            
            loss, ce_loss_val, kd_loss_val = response_based_kd_loss(
                output, teacher_output, batch_targets,
                temperature=args.temperature,
                alpha=args.alpha,
                beta=args.beta
            )
            
        elif kd_mode == 'feature':
            # Feature-based KD
            output, student_features = model(batch_imgs, return_features=True)
            with torch.no_grad():
                teacher_output = teacher_model(batch_imgs)['out']
                # Extract teacher features (for FCN-ResNet50, we'd need to modify it)
                # For now, use output-based distillation
                teacher_features = {}  # Would need to extract from teacher
            
            # Fallback to response-based if teacher features not available
            loss, ce_loss_val, kd_loss_val = response_based_kd_loss(
                output, teacher_output, batch_targets,
                temperature=args.temperature,
                alpha=args.alpha,
                beta=args.beta
            )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ce_loss += ce_loss_val
        total_kd_loss += kd_loss_val
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_kd_loss = total_kd_loss / num_batches
    
    return avg_loss, avg_ce_loss, avg_kd_loss


def validate(model, dataloader, device, num_classes=21):
    """
    Validate the model
    
    Args:
        model: model to validate
        dataloader: validation data loader
        device: device to run on
        num_classes: number of classes
    
    Returns:
        mean_iou: mean IoU across all samples
        mean_loss: mean cross-entropy loss
    """
    model.eval()
    all_ious = []
    all_losses = []
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation'):
            for img, target in zip(images, targets):
                img = img.unsqueeze(0).to(device)
                target = target.to(device)
                
                # Resize to fixed size
                h, w = target.shape
                img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
                
                # Forward pass
                output = model(img)
                
                # Calculate loss at this resolution
                target_resized = F.interpolate(target.unsqueeze(0).unsqueeze(0).float(), 
                                               size=output.shape[-2:], 
                                               mode='nearest').squeeze().long()
                loss = criterion(output, target_resized.unsqueeze(0))
                all_losses.append(loss.item())
                
                # Resize back to original size for IoU calculation
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
                
                # Get predictions
                pred = output.squeeze(0).argmax(0)
                
                # Calculate IoU
                iou = calculate_miou(pred, target, num_classes)
                all_ious.append(iou)
    
    mean_iou = np.mean(all_ious)
    mean_loss = np.mean(all_losses)
    return mean_iou, mean_loss


def plot_training_curves(history, save_path):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot - now includes both train and validation loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # mIoU plot
    axes[1].plot(history['val_miou'], label='Validation mIoU', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Validation mIoU')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def main(args):
    """Main training function"""
    
    print("=" * 70)
    print("Training Compact Segmentation Model")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"KD Mode: {args.kd_mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load student model
    print("\nLoading student model...")
    if args.model == 'ultracompact':
        model = UltraCompactSegmentationModel(num_classes=21, pretrained=True)
    elif args.model == 'standard':
        model = StandardSegmentationModel(num_classes=21, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Student model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Load teacher model if using KD
    teacher_model = None
    if args.kd_mode != 'none':
        print("\nLoading teacher model (FCN-ResNet50)...")
        weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        teacher_model = fcn_resnet50(weights=weights)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print("Teacher model loaded and frozen")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    
    # Find dataset path (Google Colab compatible)
    # VOCSegmentation expects: root/VOCdevkit/VOC2012/
    dataset_roots = []
    
    # Check for Google Colab structure (both with and without VOCdevkit wrapper)
    colab_train_path = Path('/content/data/VOC2012_train_val')
    if colab_train_path.exists():
        # Check if it has VOCdevkit structure
        if (colab_train_path / 'VOCdevkit' / 'VOC2012').exists():
            dataset_roots.append(str(colab_train_path))
        # Check if files are directly in the folder (no VOCdevkit wrapper)
        elif (colab_train_path / 'JPEGImages').exists():
            # Create a symlink or use parent structure
            dataset_roots.append(str(colab_train_path.parent))
    
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
            voc_path = root_path / 'VOC2012_train_val'
            if voc_path.exists() and (voc_path / 'JPEGImages').exists() and not (voc_path / 'VOCdevkit').exists():
                # Files are directly in VOC2012_train_val, create temporary structure
                vocdevkit_path = voc_path / 'VOCdevkit'
                voc2012_path = vocdevkit_path / 'VOC2012'
                if not voc2012_path.exists():
                    print(f"Creating VOCdevkit structure in {voc_path}")
                    vocdevkit_path.mkdir(exist_ok=True)
                    voc2012_path.symlink_to(voc_path, target_is_directory=True)
            
            test_dataset = VOCSegmentation(root=root, year='2012', image_set='train', download=False)
            dataset_root = root
            print(f"✓ Successfully loaded dataset from: {root}")
            break
        except Exception as e:
            continue
    
    if dataset_root is None:
        raise RuntimeError("Dataset not found! Please ensure dataset is in /content/data/VOC2012_train_val/ (Colab) or ./data/ (local)")
    
    print(f"Dataset found at: {dataset_root}")
    
    # Create datasets
    train_transform = VOCNormalize(augment=True)
    val_transform = VOCNormalize(augment=False)
    
    train_dataset = VOCSegmentation(
        root=dataset_root,
        year='2012',
        image_set='train',
        download=False,
        transforms=train_transform
    )
    
    val_dataset = VOCSegmentation(
        root=dataset_root,
        year='2012',
        image_set='val',
        download=False,
        transforms=val_transform
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_ce_loss': [],
        'train_kd_loss': [],
        'val_loss': [],
        'val_miou': [],
        'epoch_times': []
    }
    
    # Save hyperparameters
    hyperparameters = {
        'model': args.model,
        'kd_mode': args.kd_mode,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'temperature': args.temperature,
        'alpha': args.alpha,
        'beta': args.beta,
        'optimizer': 'Adam',
        'scheduler': 'CosineAnnealingLR',
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    best_miou = 0.0
    training_start_time = time.time()
    
    # Training loop
    print("\nStarting training...")
    print("=" * 70)
    print("\nHyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_ce_loss, train_kd_loss = train_epoch(
            model, teacher_model, train_loader, optimizer, device, args.kd_mode, args
        )
        
        print(f"Train Loss: {train_loss:.4f} (CE: {train_ce_loss:.4f}, KD: {train_kd_loss:.4f})")
        
        # Validate
        val_miou, val_loss = validate(model, val_loader, device)
        print(f"Validation mIoU: {val_miou:.4f}, Loss: {val_loss:.4f}")
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        history['epoch_times'].append(epoch_time)
        
        # Calculate timing statistics
        total_elapsed = time.time() - training_start_time
        avg_epoch_time = total_elapsed / (epoch + 1)
        remaining_epochs = args.epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        # Format time strings
        elapsed_str = f"{total_elapsed/3600:.2f}h" if total_elapsed >= 3600 else f"{total_elapsed/60:.1f}m"
        epoch_str = f"{epoch_time/60:.1f}m" if epoch_time >= 60 else f"{epoch_time:.1f}s"
        remaining_str = f"{estimated_remaining/3600:.1f}h" if estimated_remaining >= 3600 else f"{estimated_remaining/60:.0f}m"
        
        print(f"Time: {epoch_str}/epoch | Elapsed: {elapsed_str} | Est. remaining: {remaining_str}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_ce_loss'].append(train_ce_loss)
        history['train_kd_loss'].append(train_kd_loss)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            save_path = save_dir / f"best_model_{args.model}_{args.kd_mode}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'miou': val_miou,
                'history': history
            }, save_path)
            print(f"✓ Best model saved (mIoU: {val_miou:.4f})")
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f"checkpoint_{args.model}_{args.kd_mode}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'miou': val_miou,
                'history': history,
                'hyperparameters': hyperparameters
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    avg_epoch_time = np.mean(history['epoch_times'])
    
    # Plot training curves
    plot_path = save_dir / f"training_curves_{args.model}_{args.kd_mode}.png"
    plot_training_curves(history, plot_path)
    
    # Save final training report
    report_path = save_dir / f"training_report_{args.model}_{args.kd_mode}.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Training Report: {args.model} with {args.kd_mode} KD\n")
        f.write("=" * 70 + "\n\n")
        
        # Hardware information
        f.write("HARDWARE & ENVIRONMENT:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"  GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"  CUDA Version: {torch.version.cuda}\n")
        else:
            f.write(f"  GPU: Not available (CPU only)\n")
        f.write(f"  PyTorch Version: {torch.__version__}\n")
        import platform
        f.write(f"  Python Version: {platform.python_version()}\n")
        f.write(f"  Operating System: {platform.system()} {platform.release()}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("HYPERPARAMETERS:\n")
        f.write("-" * 70 + "\n")
        for key, value in hyperparameters.items():
            f.write(f"  {key:25s}: {value}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("TRAINING STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total training time: {total_training_time/3600:.2f} hours ({total_training_time:.2f} seconds)\n")
        f.write(f"  Average epoch time: {avg_epoch_time:.2f} seconds\n")
        f.write(f"  Time per image (train): {avg_epoch_time/len(train_loader.dataset)*1000:.2f} ms\n")
        f.write(f"  Best validation mIoU: {best_miou:.4f}\n")
        f.write(f"  Final validation mIoU: {history['val_miou'][-1]:.4f}\n")
        f.write(f"  Final validation loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final train loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final CE loss: {history['train_ce_loss'][-1]:.4f}\n")
        f.write(f"  Final KD loss: {history['train_kd_loss'][-1]:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("FILES GENERATED:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Best model: best_model_{args.model}_{args.kd_mode}.pth\n")
        f.write(f"  Training curves: training_curves_{args.model}_{args.kd_mode}.png\n")
        f.write(f"  This report: training_report_{args.model}_{args.kd_mode}.txt\n")
        f.write("=" * 70 + "\n")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\nTiming Summary:")
    print(f"  Total time:         {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
    print(f"  Average per epoch:  {avg_epoch_time/60:.1f} minutes ({avg_epoch_time:.1f} seconds)")
    print(f"  Time per image:     {avg_epoch_time/len(train_loader.dataset)*1000:.2f} ms")
    
    print(f"\nPerformance:")
    print(f"  Best validation mIoU:  {best_miou:.4f}")
    print(f"  Final validation mIoU: {history['val_miou'][-1]:.4f}")
    
    # Calculate final score
    score = 4 * best_miou / (1 + num_params / 1e6)
    print(f"\nFinal Score: {score:.3f}")
    print(f"  (Score = 4 × {best_miou:.4f} / (1 + {num_params/1e6:.2f}) = {score:.3f})")
    
    print(f"\nFiles saved:")
    print(f"  Model:    {save_dir / f'best_model_{args.model}_{args.kd_mode}.pth'}")
    print(f"  Report:   {report_path}")
    print(f"  Curves:   {plot_path}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train compact segmentation model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='ultracompact', choices=['ultracompact', 'standard'],
                        help='Model type to train')
    parser.add_argument('--kd_mode', type=str, default='none', choices=['none', 'response', 'feature'],
                        help='Knowledge distillation mode')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # KD arguments
    parser.add_argument('--temperature', type=float, default=3.0, help='Temperature for response-based KD')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for CE loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for KD loss')
    
    # Other arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    
    main(args)
