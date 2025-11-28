"""
Simplified COCO dataset loader for Kaggle that uses cached text embeddings.
Compatible with embeddings cached by cache_text_embeddings.py
Uses ALL 5 captions per image for 5x more training data.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os


class COCOCachedDataset(Dataset):
    """Dataset that uses pre-cached text embeddings from cache_text_embeddings.py"""
    
    def __init__(self, images_dir, embeddings_file, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Load cached embeddings
        print(f"Loading embeddings from {embeddings_file}...")
        embeddings_cache = torch.load(embeddings_file)
        
        # embeddings_cache is a dict: {"image_id_caption_idx": tensor, ...}
        # We use ALL captions per image (not just first) for more training data
        
        # Store all embeddings with their image_id and caption index
        all_embeddings = []
        all_image_ids = []
        all_caption_indices = []
        
        for key, embedding in embeddings_cache.items():
            # Key format: "image_id_caption_idx" (e.g., "391895_0")
            image_id_str, caption_idx = key.rsplit('_', 1)
            image_id = int(image_id_str)
            
            # Use ALL captions (not just caption_idx == 0)
            all_embeddings.append(embedding)
            all_image_ids.append(image_id)
            all_caption_indices.append(int(caption_idx))
        
        # Build image paths and filter out missing images
        split_name = 'train' if 'train' in str(images_dir) else 'val'
        valid_image_ids = []
        valid_embeddings = []
        valid_paths = []
        valid_caption_indices = []
        missing_count = 0
        
        for i, (img_id, caption_idx) in enumerate(zip(all_image_ids, all_caption_indices)):
            img_path = self.images_dir / f"COCO_{split_name}2014_{img_id:012d}.jpg"
            
            # Only include if image file exists
            if img_path.exists():
                valid_image_ids.append(img_id)
                valid_embeddings.append(all_embeddings[i])
                valid_paths.append(img_path)
                valid_caption_indices.append(caption_idx)
            else:
                missing_count += 1
        
        self.image_ids = valid_image_ids
        self.embeddings = torch.stack(valid_embeddings) if valid_embeddings else torch.empty(0)
        self.image_paths = valid_paths
        self.caption_indices = valid_caption_indices
        
        print(f"✓ Loaded {len(self)} samples with embeddings")
        if missing_count > 0:
            print(f"  ⚠ Skipped {missing_count} samples with missing images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get cached text embedding
        text_embedding = self.embeddings[idx]
        
        return {
            'image': image,
            'text_embedding': text_embedding,
            'image_id': self.image_ids[idx]
        }


def create_dataloaders(config=None, data_root=None, train_images_dir=None, 
                       val_images_dir=None, cache_dir=None, batch_size=64,
                       num_workers=2, pin_memory=True, **kwargs):
    """
    Create dataloaders using cached embeddings.
    Accepts both Config object and individual arguments for compatibility.
    """
    
    # Handle Config object or individual arguments
    if config is not None and hasattr(config, 'train_images_path'):
        train_images_path = config.train_images_path
        val_images_path = config.val_images_path
        cache_path = config.cache_path
        batch_size = config.batch_size
        num_workers = config.num_workers
        pin_memory = config.pin_memory
        image_size = config.image_size
        clip_mean = config.clip_mean
        clip_std = config.clip_std
    else:
        # Use individual arguments
        train_images_path = Path(data_root) / (train_images_dir or "images/train2014")
        val_images_path = Path(data_root) / (val_images_dir or "images/val2014")
        cache_path = Path(cache_dir) if cache_dir else Path("/kaggle/input/elec-475-lab4")
        image_size = 224
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
    
    # CLIP preprocessing
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std)
    ])
    
    # Create datasets
    train_dataset = COCOCachedDataset(
        images_dir=train_images_path,
        embeddings_file=cache_path / "text_embeddings_train.pt",
        transform=transform
    )
    
    val_dataset = COCOCachedDataset(
        images_dir=val_images_path,
        embeddings_file=cache_path / "text_embeddings_val.pt",
        transform=transform
    )
    
    # Apply subset if requested
    use_subset = kwargs.get('use_subset', False) or (config and getattr(config, 'use_subset', False))
    subset_size = kwargs.get('subset_size', 10000) or (config and getattr(config, 'subset_size', 10000))
    
    if use_subset:
        print(f"Subsetting validation set to {subset_size} samples...")
        if len(val_dataset) > subset_size:
            indices = torch.randperm(len(val_dataset))[:subset_size]
            val_dataset = torch.utils.data.Subset(val_dataset, indices)
            print(f"✓ Validation subset created: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
