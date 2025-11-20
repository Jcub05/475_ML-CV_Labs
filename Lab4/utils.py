"""
Utility functions for CLIP training project.
"""

import os
import json
import time
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def count_parameters(model):
    """Count trainable and total parameters in a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_json(data: Dict, filepath: Path):
    """Save dictionary as JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Dict:
    """Load JSON file as dictionary."""
    with open(filepath, 'r') as f:
        return json.load(f)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class Logger:
    """Simple logger for training progress."""
    
    def __init__(self, log_file: Optional[Path] = None, verbose: bool = True):
        self.log_file = log_file
        self.verbose = verbose
        
        if self.log_file:
            # Create parent directory if needed
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            # Clear existing log file
            with open(self.log_file, 'w') as f:
                f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
    
    def log(self, message: str):
        """Log a message to file and/or console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        if self.verbose:
            print(formatted_message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted_message + "\n")
    
    def log_dict(self, data: Dict, prefix: str = ""):
        """Log a dictionary of key-value pairs."""
        for key, value in data.items():
            if isinstance(value, float):
                self.log(f"{prefix}{key}: {value:.6f}")
            else:
                self.log(f"{prefix}{key}: {value}")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
    title: str = "Training and Validation Loss"
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_recall_metrics(
    metrics: Dict[str, float],
    save_path: Path,
    title: str = "Retrieval Performance"
):
    """
    Plot Recall@K metrics.
    
    Args:
        metrics: Dictionary with keys like 'img2txt_r1', 'img2txt_r5', etc.
        save_path: Path to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Image to Text
    img2txt_recalls = [metrics['img2txt_r1'], metrics['img2txt_r5'], metrics['img2txt_r10']]
    k_values = [1, 5, 10]
    
    ax1.bar(k_values, img2txt_recalls, color='steelblue', alpha=0.8)
    ax1.set_xlabel('K', fontsize=12)
    ax1.set_ylabel('Recall@K (%)', fontsize=12)
    ax1.set_title('Image → Text Retrieval', fontsize=13)
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(img2txt_recalls):
        ax1.text(k_values[i], v + 2, f'{v:.1f}%', ha='center', fontsize=10)
    
    # Text to Image
    txt2img_recalls = [metrics['txt2img_r1'], metrics['txt2img_r5'], metrics['txt2img_r10']]
    
    ax2.bar(k_values, txt2img_recalls, color='coral', alpha=0.8)
    ax2.set_xlabel('K', fontsize=12)
    ax2.set_ylabel('Recall@K (%)', fontsize=12)
    ax2.set_title('Text → Image Retrieval', fontsize=13)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(txt2img_recalls):
        ax2.text(k_values[i], v + 2, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: Path,
    is_best: bool = False
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = filepath.parent / "best_model.pth"
        torch.save(checkpoint, best_path)


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda"
):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: PyTorch optimizer (optional)
        device: Device to load model to
        
    Returns:
        Dictionary with epoch, loss, and metrics
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
        'metrics': checkpoint.get('metrics', {})
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test timer
    with Timer() as t:
        time.sleep(0.1)
    print(f"Timer test: {format_time(t.elapsed)}")
    
    # Test average meter
    meter = AverageMeter("Loss")
    for i in range(10):
        meter.update(i)
    print(f"Average meter test: {meter}")
    
    # Test logger
    logger = Logger(verbose=True)
    logger.log("Test log message")
    logger.log_dict({"accuracy": 0.95, "loss": 0.05}, prefix="Test ")
    
    print("✓ Utilities test complete")
