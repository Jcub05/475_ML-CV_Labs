"""
COCO 2014 Dataset Loader for CLIP Training.
Handles image loading, caption processing, and text embedding caching.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as transforms


class COCOImageCaptionDataset(Dataset):
    """
    COCO 2014 Dataset for image-caption pairs.
    Supports loading pre-cached text embeddings for efficiency.
    """
    
    def __init__(
        self,
        image_dir: Path,
        captions_file: Path,
        transform: Optional[transforms.Compose] = None,
        cached_embeddings_path: Optional[Path] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize COCO dataset.
        
        Args:
            image_dir: Directory containing images (train2014/ or val2014/)
            captions_file: Path to captions JSON file
            transform: Image transformations to apply
            cached_embeddings_path: Path to cached text embeddings file (.pt)
            max_samples: Maximum number of samples to use (for subset experiments)
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.cached_embeddings_path = cached_embeddings_path
        
        # Load captions from JSON
        print(f"Loading captions from {captions_file}...")
        with open(captions_file, 'r') as f:
            coco_data = json.load(f)
        
        # Parse annotations
        self.images_info = {img['id']: img for img in coco_data['images']}
        
        # Group captions by image_id
        self.image_id_to_captions = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            
            if image_id not in self.image_id_to_captions:
                self.image_id_to_captions[image_id] = []
            self.image_id_to_captions[image_id].append(caption)
        
        # Create list of (image_id, caption_idx) pairs
        self.samples = []
        for image_id, captions in self.image_id_to_captions.items():
            for caption_idx in range(len(captions)):
                self.samples.append((image_id, caption_idx))
        
        # Apply max_samples limit if specified
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
            print(f"Using subset of {max_samples} samples")
        
        # Load cached embeddings if available
        self.cached_embeddings = None
        if cached_embeddings_path and cached_embeddings_path.exists():
            print(f"Loading cached text embeddings from {cached_embeddings_path}...")
            self.cached_embeddings = torch.load(cached_embeddings_path)
            print(f"✓ Loaded {len(self.cached_embeddings)} cached embeddings")
        
        print(f"Dataset initialized with {len(self.samples)} image-caption pairs")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - image: Transformed image tensor [3, H, W]
                - text_embedding: Text embedding tensor [embed_dim] (if cached)
                - caption: Raw caption string
                - image_id: Image ID
                - image_path: Path to image file
        """
        image_id, caption_idx = self.samples[idx]
        
        # Get image info
        image_info = self.images_info[image_id]
        image_filename = image_info['file_name']
        image_path = self.image_dir / image_filename
        
        # Load and transform image
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            if self.transform:
                image = self.transform(image)
        
        # Get caption
        caption = self.image_id_to_captions[image_id][caption_idx]
        
        # Prepare return dict
        sample = {
            'image': image,
            'caption': caption,
            'image_id': image_id,
            'image_path': str(image_path)
        }
        
        # Add cached text embedding if available
        if self.cached_embeddings is not None:
            # Key format: "image_id_caption_idx"
            embedding_key = f"{image_id}_{caption_idx}"
            if embedding_key in self.cached_embeddings:
                sample['text_embedding'] = self.cached_embeddings[embedding_key]
            else:
                # Fallback: return zero embedding if not found
                sample['text_embedding'] = torch.zeros(512)
        
        return sample


def get_clip_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Get CLIP preprocessing transforms.
    
    Args:
        image_size: Target image size (default: 224)
        is_train: Whether to apply training augmentations
        
    Returns:
        Composed transforms
    """
    # CLIP normalization values
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    
    if is_train:
        # Training transforms with light augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std)
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std)
        ])
    
    return transform


def create_dataloaders(
    data_root: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_cached_embeddings: bool = True,
    use_subset: bool = False,
    subset_size: int = 10000
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        data_root: Root directory of COCO dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        use_cached_embeddings: Whether to use cached text embeddings
        use_subset: Whether to use a subset of data
        subset_size: Size of subset if use_subset=True
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_root = Path(data_root)
    
    # Define paths
    train_images_dir = data_root / "train2014"
    val_images_dir = data_root / "val2014"
    train_captions_file = data_root / "annotations" / "captions_train2014.json"
    val_captions_file = data_root / "annotations" / "captions_val2014.json"
    
    # Cached embeddings paths
    cache_dir = data_root / "cached_embeddings"
    train_cache = cache_dir / "text_embeddings_train.pt" if use_cached_embeddings else None
    val_cache = cache_dir / "text_embeddings_val.pt" if use_cached_embeddings else None
    
    # Get transforms
    train_transform = get_clip_transforms(is_train=True)
    val_transform = get_clip_transforms(is_train=False)
    
    # Create datasets
    print("\n" + "=" * 80)
    print("Creating Training Dataset")
    print("=" * 80)
    train_dataset = COCOImageCaptionDataset(
        image_dir=train_images_dir,
        captions_file=train_captions_file,
        transform=train_transform,
        cached_embeddings_path=train_cache,
        max_samples=subset_size if use_subset else None
    )
    
    print("\n" + "=" * 80)
    print("Creating Validation Dataset")
    print("=" * 80)
    val_dataset = COCOImageCaptionDataset(
        image_dir=val_images_dir,
        captions_file=val_captions_file,
        transform=val_transform,
        cached_embeddings_path=val_cache,
        max_samples=subset_size // 5 if use_subset else None  # Smaller val set
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print("\n" + "=" * 80)
    print("DataLoader Summary")
    print("=" * 80)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {batch_size}")
    print()
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    from config import get_config
    
    print("Testing COCO Dataset Loader\n")
    
    # Get config
    config = get_config()
    
    try:
        # Test transforms
        print("Testing transforms...")
        train_transform = get_clip_transforms(is_train=True)
        val_transform = get_clip_transforms(is_train=False)
        print("✓ Transforms created successfully\n")
        
        # Test dataset creation
        print("Testing dataset creation...")
        train_loader, val_loader = create_dataloaders(
            data_root=config.data_root,
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            use_cached_embeddings=False,  # Don't require cache for testing
            use_subset=True,
            subset_size=100
        )
        
        # Test loading a batch
        print("Testing batch loading...")
        batch = next(iter(train_loader))
        
        print(f"\nBatch contents:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Number of captions: {len(batch['caption'])}")
        print(f"  Image IDs: {batch['image_id'][:3]}...")
        
        if 'text_embedding' in batch:
            print(f"  Text embedding shape: {batch['text_embedding'].shape}")
        
        print(f"\nSample caption: {batch['caption'][0][:100]}...")
        
        print("\n✓ Dataset test complete!")
        
    except Exception as e:
        print(f"\n✗ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
