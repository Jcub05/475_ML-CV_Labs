"""
Configuration file for CLIP fine-tuning on COCO 2014 dataset.
Supports both local (Windows) and Colab environments.
"""

import os
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class Config:
    """Configuration class for CLIP training and evaluation."""
    
    # ========== Environment Detection ==========
    # Automatically detect if running on Colab
    is_colab: bool = 'COLAB_GPU' in os.environ
    
    # ========== Data Paths ==========
    # Base path to dataset (outside git repo)
    data_root: str = None  # Will be set based on environment
    
    # Dataset subdirectories
    train_images_dir: str = "train2014"
    val_images_dir: str = "val2014"
    train_captions_file: str = "annotations/captions_train2014.json"
    val_captions_file: str = "annotations/captions_val2014.json"
    
    # Cache directory for text embeddings
    cache_dir: str = "cached_embeddings"
    
    # Checkpoint directory
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    
    # ========== Model Architecture ==========
    embed_dim: int = 512
    image_size: int = 224
    pretrained_resnet: bool = True
    
    # CLIP preprocessing constants
    clip_mean: tuple = (0.48145466, 0.4578275, 0.40821073)
    clip_std: tuple = (0.26862954, 0.26130258, 0.27577711)
    
    # HuggingFace CLIP text encoder
    clip_model_name: str = "openai/clip-vit-base-patch32"
    
    # ========== Training Hyperparameters ==========
    batch_size: int = 64
    num_epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine" or "reduce_on_plateau"
    
    # ========== Optimization ==========
    optimizer_type: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # ========== Hardware ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Mixed precision training
    use_amp: bool = True
    
    # ========== Data Loading ==========
    # Use subset of data for faster experimentation
    use_subset: bool = False
    subset_size: int = 10000
    
    # Whether to cache text embeddings
    use_cached_embeddings: bool = True
    
    # ========== Evaluation ==========
    eval_every_n_epochs: int = 1
    save_best_only: bool = True
    recall_k_values: list = None  # Will be set to [1, 5, 10]
    
    # ========== Visualization ==========
    num_visualization_samples: int = 10
    save_visualizations: bool = True
    
    # ========== Logging ==========
    log_interval: int = 100  # Log every N batches
    verbose: bool = True
    
    def __post_init__(self):
        """Initialize derived paths based on environment."""
        # Set recall K values
        if self.recall_k_values is None:
            self.recall_k_values = [1, 5, 10]
        
        # Set data root based on environment
        if self.data_root is None:
            if self.is_colab:
                # Colab environment - assuming Google Drive is mounted
                self.data_root = "/content/drive/MyDrive/datasets_Lab4/coco_2014"
            else:
                # Local Windows environment
                self.data_root = r"C:\Users\jcube\OneDrive\Desktop\Jacob\School\Queens\Year 5\ELEC 475\datasets_Lab4\coco_2014"
        
        # Convert to Path objects for easier manipulation
        self.data_root = Path(self.data_root)
        
        # Full paths
        self.train_images_path = self.data_root / self.train_images_dir
        self.val_images_path = self.data_root / self.val_images_dir
        self.train_captions_path = self.data_root / self.train_captions_file
        self.val_captions_path = self.data_root / self.val_captions_file
        self.cache_path = self.data_root / self.cache_dir
        
        # Create checkpoint and results directories
        # On Colab: save to Google Drive (persistent)
        # On Local: save to Lab4 folder
        if self.is_colab:
            # Save to Drive for persistence across sessions
            drive_base = Path("/content/drive/MyDrive/datasets_Lab4")
            self.checkpoint_path = drive_base / "Lab4_checkpoints"
            self.results_path = drive_base / "Lab4_results"
        else:
            # Local: save in Lab4 folder
            lab4_dir = Path(__file__).parent
            self.checkpoint_path = lab4_dir / self.checkpoint_dir
            self.results_path = lab4_dir / self.results_dir
        
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        
    def validate_paths(self):
        """Validate that required dataset paths exist."""
        required_paths = [
            self.train_images_path,
            self.val_images_path,
            self.train_captions_path,
            self.val_captions_path
        ]
        
        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            raise FileNotFoundError(
                f"Required dataset paths not found:\n" + 
                "\n".join(f"  - {p}" for p in missing_paths) +
                f"\n\nPlease download the dataset using download_dataset.py"
            )
        
        return True
    
    def __repr__(self):
        """String representation of config."""
        lines = ["Configuration:"]
        lines.append(f"  Environment: {'Colab' if self.is_colab else 'Local'}")
        lines.append(f"  Device: {self.device}")
        lines.append(f"  Data Root: {self.data_root}")
        lines.append(f"  Batch Size: {self.batch_size}")
        lines.append(f"  Learning Rate: {self.learning_rate}")
        lines.append(f"  Epochs: {self.num_epochs}")
        lines.append(f"  Use Cached Embeddings: {self.use_cached_embeddings}")
        lines.append(f"  Mixed Precision: {self.use_amp}")
        return "\n".join(lines)


def get_config(**kwargs):
    """
    Factory function to create Config with custom parameters.
    
    Args:
        **kwargs: Override default config values
        
    Returns:
        Config object
    """
    config = Config(**kwargs)
    config.create_directories()
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print(config)
    print("\nValidating paths...")
    try:
        config.validate_paths()
        print("✓ All paths valid")
    except FileNotFoundError as e:
        print(f"✗ {e}")
