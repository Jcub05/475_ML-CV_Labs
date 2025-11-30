"""
Kaggle-specific configuration for CLIP fine-tuning.
This config detects Kaggle environment and sets appropriate paths.
"""

import os
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class Config:
    """Configuration class for CLIP training on Kaggle."""
    
    # ========== Environment Detection ==========
    is_kaggle: bool = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    is_colab: bool = 'COLAB_GPU' in os.environ
    
    # ========== Data Paths ==========
    data_root: str = None  # Will be set based on environment
    
    # Dataset subdirectories (Kaggle COCO dataset structure)
    train_images_dir: str = "images/train2014"
    val_images_dir: str = "images/val2014"
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
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    
    # ========== Optimization ==========
    optimizer_type: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # ========== Hardware ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2  # Kaggle: use fewer workers
    pin_memory: bool = True
    
    # Mixed precision training
    use_amp: bool = True
    
    # ========== Data Loading ==========
    use_subset: bool = False
    subset_size: int = 10000
    
    # Whether to cache text embeddings
    use_cached_embeddings: bool = True
    
    # ========== Evaluation ==========
    eval_every_n_epochs: int = 1
    save_best_only: bool = False  # Save all checkpoints on Kaggle
    recall_k_values: list = None
    
    # ========== Visualization ==========
    num_visualization_samples: int = 10
    save_visualizations: bool = True
    
    # ========== Logging ==========
    log_interval: int = 100
    verbose: bool = True
    
    def __post_init__(self):
        """Initialize derived paths based on environment."""
        if self.recall_k_values is None:
            self.recall_k_values = [1, 5, 10]
        
        # Set data root based on environment
        if self.data_root is None:
            if self.is_kaggle:
                # Kaggle environment - datasets are in /kaggle/input/
                self.data_root = "/kaggle/input/coco-2014-dataset-for-yolov3"
                self.text_embeddings_path = "/kaggle/input/elec-475-lab4"
            elif self.is_colab:
                self.data_root = "/content/drive/MyDrive/datasets_Lab4/coco_2014"
            else:
                # Local Windows environment
                self.data_root = r"C:\Users\jcube\OneDrive\Desktop\Jacob\School\Queens\Year 5\ELEC 475\datasets_Lab4\coco_2014"
        
        # Convert to Path objects
        self.data_root = Path(self.data_root)
        
        # Full paths
        self.train_images_path = self.data_root / self.train_images_dir
        self.val_images_path = self.data_root / self.val_images_dir
        self.train_captions_path = self.data_root / self.train_captions_file
        self.val_captions_path = self.data_root / self.val_captions_file
        
        # Cache path - use text embeddings dataset on Kaggle
        if self.is_kaggle and hasattr(self, 'text_embeddings_path'):
            self.cache_path = Path(self.text_embeddings_path)
        else:
            self.cache_path = self.data_root / self.cache_dir
        
        # Checkpoint and results directories
        if self.is_kaggle:
            # Kaggle: save to /kaggle/working/ (downloadable outputs)
            self.checkpoint_path = Path("/kaggle/working") / self.checkpoint_dir
            self.results_path = Path("/kaggle/working") / self.results_dir
        elif self.is_colab:
            drive_base = Path("/content/drive/MyDrive/datasets_Lab4")
            self.checkpoint_path = drive_base / "Lab4_checkpoints"
            self.results_path = drive_base / "Lab4_results"
        else:
            # Local
            lab4_dir = Path(__file__).parent
            self.checkpoint_path = lab4_dir / self.checkpoint_dir
            self.results_path = lab4_dir / self.results_dir
        
    def create_directories(self):
        """Create necessary directories if they don't exist."""
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
                "\n".join(f"  - {p}" for p in missing_paths)
            )
        
        return True
    
    def __repr__(self):
        """String representation of config."""
        lines = ["Configuration:"]
        if self.is_kaggle:
            lines.append("  Environment: Kaggle")
        elif self.is_colab:
            lines.append("  Environment: Colab")
        else:
            lines.append("  Environment: Local")
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
