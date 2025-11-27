"""
Training script for CLIP fine-tuning on COCO 2014 dataset.
Version: 2.0 - Memory optimized (no Recall@K in validation)
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm

from config import get_config
from dataset import create_dataloaders
from model import CLIPFineTuneModel, count_parameters
from loss import InfoNCELossWithMetrics
from metrics import compute_retrieval_metrics
from utils import (
    set_seed, get_device, AverageMeter, Timer, Logger,
    plot_training_curves, plot_recall_metrics, save_checkpoint, format_time
)


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: str,
    epoch: int,
    config,
    logger: Logger,
    scaler=None
):
    """
    Train for one epoch.
    
    Args:
        model: CLIP model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        config: Configuration object
        logger: Logger instance
        scaler: GradScaler for mixed precision (optional)
        
    Returns:
        Dictionary with epoch metrics
    """
    model.train()
    
    # Meters for tracking
    loss_meter = AverageMeter("Loss")
    i2t_acc_meter = AverageMeter("I2T_Acc")
    t2i_acc_meter = AverageMeter("T2I_Acc")
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].to(device)
        text_embeddings = batch['text_embedding'].to(device)
        
        batch_size = images.size(0)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if config.use_amp and scaler is not None:
            with autocast():
                image_embeds, text_embeds = model(images, text_embeddings=text_embeddings)
                loss, metrics = criterion(image_embeds, text_embeds, return_metrics=True)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular forward pass
            image_embeds, text_embeds = model(images, text_embeddings=text_embeddings)
            loss, metrics = criterion(image_embeds, text_embeds, return_metrics=True)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
        
        # Update meters
        loss_meter.update(metrics['loss'], batch_size)
        i2t_acc_meter.update(metrics['i2t_acc'], batch_size)
        t2i_acc_meter.update(metrics['t2i_acc'], batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_meter.avg:.4f}",
            'i2t': f"{i2t_acc_meter.avg:.1f}%",
            't2i': f"{t2i_acc_meter.avg:.1f}%"
        })
        
        # Log periodically
        if (batch_idx + 1) % config.log_interval == 0:
            logger.log(
                f"Epoch [{epoch}/{config.num_epochs}] "
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss_meter.avg:.4f} "
                f"I2T: {i2t_acc_meter.avg:.2f}% "
                f"T2I: {t2i_acc_meter.avg:.2f}%"
            )
    
    return {
        'loss': loss_meter.avg,
        'i2t_acc': i2t_acc_meter.avg,
        't2i_acc': t2i_acc_meter.avg
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device: str,
    config,
    logger: Logger
):
    """
    Validate for one epoch.
    
    Args:
        model: CLIP model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device
        config: Configuration
        logger: Logger
        
    Returns:
        Dictionary with validation loss only (Recall@K computed separately)
    """
    model.eval()
    
    loss_meter = AverageMeter("Val_Loss")
    
    pbar = tqdm(dataloader, desc="Validation")
    
    for batch in pbar:
        images = batch['image'].to(device)
        text_embeddings = batch['text_embedding'].to(device)
        
        batch_size = images.size(0)
        
        # Forward pass
        if config.use_amp:
            with autocast():
                image_embeds, text_embeds = model(images, text_embeddings=text_embeddings)
                loss, metrics = criterion(image_embeds, text_embeds, return_metrics=True)
        else:
            image_embeds, text_embeds = model(images, text_embeddings=text_embeddings)
            loss, metrics = criterion(image_embeds, text_embeds, return_metrics=True)
        
        loss_meter.update(metrics['loss'], batch_size)
        
        pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})
    
    # Clear GPU cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Return only loss (Recall@K will be computed separately after training)
    val_metrics = {
        'loss': loss_meter.avg
    }
    
    # Log metrics
    logger.log(f"Validation Loss: {val_metrics['loss']:.4f}")
    logger.log("(Recall@K will be computed after training completes)")
    
    return val_metrics


def train_model(config, args):
    """
    Main training loop.
    
    Args:
        config: Configuration object
        args: Command line arguments
    """
    # Setup
    set_seed(42)
    device = get_device()
    
    # Logger
    log_file = config.results_path / "training_log.txt"
    logger = Logger(log_file=log_file, verbose=config.verbose)
    
    logger.log("=" * 80)
    logger.log("CLIP Fine-tuning Training")
    logger.log("=" * 80)
    logger.log(str(config))
    
    # Create dataloaders
    logger.log("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        use_cached_embeddings=config.use_cached_embeddings,
        use_subset=config.use_subset,
        subset_size=config.subset_size
    )
    
    # Create model
    logger.log("\nCreating model...")
    model = CLIPFineTuneModel(
        embed_dim=config.embed_dim,
        pretrained_resnet=config.pretrained_resnet,
        clip_model_name=config.clip_model_name,
        freeze_text_encoder=True
    ).to(device)
    
    # Count parameters
    trainable, total = count_parameters(model)
    logger.log(f"Model parameters: {trainable:,} trainable / {total:,} total")
    logger.log(f"Trainable percentage: {100 * trainable / total:.2f}%")
    
    # Loss function
    criterion = InfoNCELossWithMetrics(
        temperature=config.temperature,
        learnable_temperature=False
    )
    
    # Optimizer
    if config.optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    logger.log(f"Optimizer: {config.optimizer_type}")
    logger.log(f"Learning rate: {config.learning_rate}")
    
    # Learning rate scheduler
    scheduler = None
    if config.use_scheduler:
        if config.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs,
                eta_min=config.learning_rate * 0.01
            )
            logger.log("Scheduler: CosineAnnealingLR")
        elif config.scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                verbose=True
            )
            logger.log("Scheduler: ReduceLROnPlateau")
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_amp else None
    if config.use_amp:
        logger.log("Using Automatic Mixed Precision (AMP)")
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Track best validation loss
    
    # Training loop
    logger.log("\n" + "=" * 80)
    logger.log("Starting Training")
    logger.log("=" * 80 + "\n")
    
    total_timer = Timer()
    total_timer.start()
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_timer = Timer()
        epoch_timer.start()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, config, logger, scaler
        )
        train_losses.append(train_metrics['loss'])
        
        # Validate
        if epoch % config.eval_every_n_epochs == 0:
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, config, logger
            )
            val_losses.append(val_metrics['loss'])
            
            # Check if best model (lowest validation loss)
            current_loss = val_metrics['loss']
            is_best = current_loss < best_val_loss
            
            if is_best:
                best_val_loss = current_loss
                logger.log(f"✓ New best model! Val Loss: {best_val_loss:.4f}")
            
            # Save checkpoint
            if not config.save_best_only or is_best:
                checkpoint_name = f"checkpoint_epoch_{epoch}.pth" if not config.save_best_only else "best_model.pth"
                checkpoint_path = config.checkpoint_path / checkpoint_name
                
                save_checkpoint(
                    model, optimizer, epoch, val_metrics['loss'],
                    val_metrics, checkpoint_path, is_best
                )
                logger.log(f"Checkpoint saved: {checkpoint_path.name}")
        
        # Learning rate scheduling
        if scheduler is not None:
            if config.scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_losses[-1])
            else:
                scheduler.step()
        
        # Log epoch summary
        epoch_time = epoch_timer.stop()
        logger.log(f"\nEpoch {epoch} completed in {format_time(epoch_time)}")
        logger.log(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}\n")
    
    # Training complete
    total_time = total_timer.stop()
    logger.log("\n" + "=" * 80)
    logger.log("Training Complete!")
    logger.log("=" * 80)
    logger.log(f"Total training time: {format_time(total_time)}")
    logger.log(f"Best validation loss: {best_val_loss:.4f}")
    logger.log("\nNote: Run evaluate.py to compute Recall@K metrics")
    
    # Plot training curves
    logger.log("\nGenerating plots...")
    
    plot_path = config.results_path / "training_curves.png"
    plot_training_curves(train_losses, val_losses, plot_path)
    logger.log(f"Training curves saved: {plot_path}")
    
    logger.log("\nAll done! Run evaluate.py to compute Recall@K metrics.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CLIP model on COCO dataset")
    
    # Data
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory of COCO dataset')
    parser.add_argument('--use_subset', action='store_true',
                        help='Use subset of data for faster experimentation')
    parser.add_argument('--subset_size', type=int, default=10000,
                        help='Subset size if use_subset=True')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Model
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for InfoNCE loss')
    
    # Other
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Get configuration with overrides from args
    config_kwargs = {}
    if args.data_root:
        config_kwargs['data_root'] = args.data_root
    if args.use_subset:
        config_kwargs['use_subset'] = True
        config_kwargs['subset_size'] = args.subset_size
    if args.batch_size:
        config_kwargs['batch_size'] = args.batch_size
    if args.num_epochs:
        config_kwargs['num_epochs'] = args.num_epochs
    if args.learning_rate:
        config_kwargs['learning_rate'] = args.learning_rate
    if args.weight_decay:
        config_kwargs['weight_decay'] = args.weight_decay
    if args.temperature:
        config_kwargs['temperature'] = args.temperature
    if args.no_amp:
        config_kwargs['use_amp'] = False
    if args.num_workers:
        config_kwargs['num_workers'] = args.num_workers
    
    config = get_config(**config_kwargs)
    
    try:
        # Validate dataset exists
        config.validate_paths()
        
        # Start training
        train_model(config, args)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
