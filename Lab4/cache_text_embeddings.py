"""
Pre-compute and cache text embeddings for COCO captions.
This significantly speeds up training by avoiding repeated text encoding.
"""

import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from config import get_config


def cache_text_embeddings(
    captions_file: Path,
    output_file: Path,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 256,
    device: str = "cuda",
    use_local_cache: bool = True
):
    """
    Pre-compute and cache text embeddings for all captions.
    
    Args:
        captions_file: Path to COCO captions JSON file
        output_file: Path to save cached embeddings (.pt file)
        model_name: HuggingFace CLIP model name
        batch_size: Batch size for encoding (larger = faster but more memory)
        device: Device to run encoding on
        use_local_cache: If True, save to /content first then copy to final location (faster on Colab)
    """
    print(f"\n{'=' * 80}")
    print(f"Caching Text Embeddings")
    print(f"{'=' * 80}")
    print(f"Captions file: {captions_file}")
    print(f"Output file: {output_file}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}\n")
    
    # For Colab: use local temp storage for faster I/O
    if use_local_cache and '/content/drive/' in str(output_file):
        temp_output = Path('/content') / 'temp_cache' / output_file.name
        temp_output.parent.mkdir(parents=True, exist_ok=True)
        print(f"⚡ Using local cache for speed: {temp_output}")
        print(f"   Will copy to Drive when complete.\n")
        actual_output = temp_output
    else:
        actual_output = output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load CLIP model and processor
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    print("✓ Model loaded\n")
    
    # Load captions
    print("Loading captions...")
    with open(captions_file, 'r') as f:
        coco_data = json.load(f)
    
    annotations = coco_data['annotations']
    print(f"✓ Loaded {len(annotations)} captions\n")
    
    # Prepare for caching
    embeddings_cache = {}
    
    # Process in batches
    print("Encoding captions...")
    num_batches = (len(annotations) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(annotations), batch_size), total=num_batches):
            batch_annotations = annotations[i:i + batch_size]
            
            # Extract captions and metadata
            batch_captions = [ann['caption'] for ann in batch_annotations]
            batch_image_ids = [ann['image_id'] for ann in batch_annotations]
            
            # Tokenize captions
            inputs = processor(
                text=batch_captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP's max sequence length
            ).to(device)
            
            # Get text embeddings
            text_features = model.get_text_features(**inputs)
            
            # Normalize embeddings (important for cosine similarity)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Move to CPU and store
            text_features = text_features.cpu()
            
            # Store embeddings with unique keys
            # We need to track which caption (by index) for each image_id
            # Since COCO has multiple captions per image
            for j, (image_id, embedding) in enumerate(zip(batch_image_ids, text_features)):
                # Count how many times we've seen this image_id
                caption_idx = sum(1 for k in embeddings_cache.keys() if k.startswith(f"{image_id}_"))
                
                # Create unique key: "image_id_caption_idx"
                key = f"{image_id}_{caption_idx}"
                embeddings_cache[key] = embedding
    
    print(f"\n✓ Encoded {len(embeddings_cache)} caption embeddings\n")
    
    # Save to disk (local temp first if using Colab)
    print(f"Saving embeddings to {actual_output}...")
    torch.save(embeddings_cache, actual_output)
    
    # Verify save
    file_size_mb = actual_output.stat().st_size / (1024 * 1024)
    print(f"✓ Saved successfully ({file_size_mb:.2f} MB)\n")
    
    # Copy to Drive if we used local cache
    if use_local_cache and actual_output != output_file:
        print(f"Copying to Google Drive: {output_file}...")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(actual_output, output_file)
        print(f"✓ Copied to Drive ({file_size_mb:.2f} MB)\n")
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
    
    # Print statistics
    print(f"{'=' * 80}")
    print("Cache Statistics")
    print(f"{'=' * 80}")
    print(f"Total embeddings: {len(embeddings_cache):,}")
    print(f"Embedding dimension: {embeddings_cache[list(embeddings_cache.keys())[0]].shape[0]}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Average per embedding: {file_size_mb / len(embeddings_cache) * 1024:.2f} KB")
    print()
    
    return embeddings_cache


def main():
    """Main function to cache embeddings for train and val sets."""
    print("\nCOCO Caption Embeddings Caching Script")
    print("This will pre-compute text embeddings for faster training.\n")
    
    # Get configuration
    config = get_config()
    
    # Check if dataset exists
    try:
        config.validate_paths()
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease run download_dataset.py first to download the COCO dataset.")
        return
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("\n⚠ Warning: Running on CPU. This will be much slower than GPU.")
        print("Consider using Google Colab with GPU for faster processing.\n")
    
    # Cache directory
    cache_dir = config.data_root / "cached_embeddings"
    cache_dir.mkdir(exist_ok=True)
    
    # Determine batch size based on device
    batch_size = 256 if device == "cuda" else 32
    
    # Cache training embeddings
    print("\n" + "=" * 80)
    print("TRAINING SET")
    print("=" * 80)
    
    train_cache_file = cache_dir / "text_embeddings_train.pt"
    
    if train_cache_file.exists():
        print(f"\n⚠ Training cache already exists: {train_cache_file}")
        response = input("Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            print("Skipping training set...")
        else:
            cache_text_embeddings(
                captions_file=config.train_captions_path,
                output_file=train_cache_file,
                model_name=config.clip_model_name,
                batch_size=batch_size,
                device=device
            )
    else:
        cache_text_embeddings(
            captions_file=config.train_captions_path,
            output_file=train_cache_file,
            model_name=config.clip_model_name,
            batch_size=batch_size,
            device=device
        )
    
    # Cache validation embeddings
    print("\n" + "=" * 80)
    print("VALIDATION SET")
    print("=" * 80)
    
    val_cache_file = cache_dir / "text_embeddings_val.pt"
    
    if val_cache_file.exists():
        print(f"\n⚠ Validation cache already exists: {val_cache_file}")
        response = input("Overwrite? (y/n): ").lower().strip()
        if response != 'y':
            print("Skipping validation set...")
        else:
            cache_text_embeddings(
                captions_file=config.val_captions_path,
                output_file=val_cache_file,
                model_name=config.clip_model_name,
                batch_size=batch_size,
                device=device
            )
    else:
        cache_text_embeddings(
            captions_file=config.val_captions_path,
            output_file=val_cache_file,
            model_name=config.clip_model_name,
            batch_size=batch_size,
            device=device
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ Caching Complete!")
    print("=" * 80)
    print("\nCached files:")
    print(f"  Training: {train_cache_file}")
    print(f"  Validation: {val_cache_file}")
    print("\nYou can now run training with cached embeddings for faster performance.")
    print("Run: python train.py")
    print()


if __name__ == "__main__":
    main()
