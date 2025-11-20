"""
Download COCO 2014 dataset using kagglehub.
Dataset will be saved to datasets_Lab4/coco_2014/ outside the git repository.
"""

import os
import sys
import shutil
from pathlib import Path
import kagglehub


def download_coco_dataset():
    """
    Download COCO 2014 dataset and organize it in the correct structure.
    
    Returns:
        Path to the organized dataset directory
    """
    print("=" * 80)
    print("COCO 2014 Dataset Download")
    print("=" * 80)
    
    # Define target directory (outside git repo)
    # Get the parent directory of the 475_ML-CV_Labs folder
    lab4_dir = Path(__file__).parent
    repo_dir = lab4_dir.parent
    base_dir = repo_dir.parent
    target_dir = base_dir / "datasets_Lab4" / "coco_2014"
    
    print(f"\nTarget directory: {target_dir}")
    
    # Check if dataset already exists
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"\n⚠ Dataset directory already exists: {target_dir}")
        response = input("Do you want to re-download? (y/n): ").lower().strip()
        if response != 'y':
            print("Using existing dataset.")
            verify_dataset_structure(target_dir)
            return target_dir
        else:
            print("Removing existing dataset...")
            shutil.rmtree(target_dir)
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Downloading dataset from Kaggle...")
    print("This may take a while (~13 GB download)")
    print("=" * 80 + "\n")
    
    try:
        # Download dataset using kagglehub
        download_path = kagglehub.dataset_download("jeffaudi/coco-2014-dataset-for-yolov3")
        download_path = Path(download_path)
        
        print(f"\n✓ Download complete!")
        print(f"Downloaded to: {download_path}")
        
        # Organize the dataset structure
        print("\n" + "=" * 80)
        print("Organizing dataset structure...")
        print("=" * 80 + "\n")
        
        # Expected files and directories in the downloaded dataset
        required_items = {
            'train2014': 'dir',
            'val2014': 'dir',
            'annotations': 'dir',  # Contains captions JSON files
        }
        
        # Copy/move files to target directory
        for item_name, item_type in required_items.items():
            source = download_path / item_name
            destination = target_dir / item_name
            
            if not source.exists():
                print(f"⚠ Warning: {item_name} not found in downloaded dataset")
                continue
            
            if destination.exists():
                print(f"  {item_name} already exists, skipping...")
                continue
            
            if item_type == 'dir':
                print(f"  Copying {item_name}...")
                shutil.copytree(source, destination)
            else:
                print(f"  Copying {item_name}...")
                shutil.copy2(source, destination)
        
        # Verify the structure
        print("\n" + "=" * 80)
        print("Verifying dataset structure...")
        print("=" * 80 + "\n")
        
        verify_dataset_structure(target_dir)
        
        print("\n" + "=" * 80)
        print("✓ Dataset download and setup complete!")
        print("=" * 80)
        print(f"\nDataset location: {target_dir}")
        print("\nYou can now proceed with training.")
        
        return target_dir
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have kagglehub installed: pip install kagglehub")
        print("  2. Ensure you have Kaggle API credentials configured")
        print("  3. Check your internet connection")
        print("  4. Verify you have enough disk space (~13 GB)")
        sys.exit(1)


def verify_dataset_structure(dataset_dir: Path):
    """
    Verify that the dataset has the correct structure.
    
    Args:
        dataset_dir: Path to the dataset directory
    """
    required_structure = {
        'train2014': 'Directory with training images',
        'val2014': 'Directory with validation images',
        'annotations/captions_train2014.json': 'Training captions',
        'annotations/captions_val2014.json': 'Validation captions',
    }
    
    all_present = True
    
    for path_str, description in required_structure.items():
        full_path = dataset_dir / path_str
        
        if full_path.exists():
            if full_path.is_dir():
                # Count files in directory
                num_files = len(list(full_path.glob('*')))
                print(f"  ✓ {path_str} ({num_files} items)")
            else:
                # Get file size
                size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {path_str} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {path_str} - MISSING")
            all_present = False
    
    if not all_present:
        print("\n⚠ Warning: Some required files are missing!")
        print("The dataset may not be complete. Training may fail.")
    else:
        print("\n✓ All required files and directories present!")
    
    return all_present


def get_dataset_stats(dataset_dir: Path):
    """
    Print statistics about the dataset.
    
    Args:
        dataset_dir: Path to the dataset directory
    """
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80 + "\n")
    
    # Count training images
    train_images = list((dataset_dir / "train2014").glob("*.jpg"))
    print(f"Training images: {len(train_images):,}")
    
    # Count validation images
    val_images = list((dataset_dir / "val2014").glob("*.jpg"))
    print(f"Validation images: {len(val_images):,}")
    
    # Load caption files and count captions
    import json
    
    train_captions_file = dataset_dir / "annotations" / "captions_train2014.json"
    if train_captions_file.exists():
        with open(train_captions_file, 'r') as f:
            train_data = json.load(f)
            print(f"Training captions: {len(train_data.get('annotations', [])):,}")
    
    val_captions_file = dataset_dir / "annotations" / "captions_val2014.json"
    if val_captions_file.exists():
        with open(val_captions_file, 'r') as f:
            val_data = json.load(f)
            print(f"Validation captions: {len(val_data.get('annotations', [])):,}")
    
    print()


if __name__ == "__main__":
    print("\nCOCO 2014 Dataset Downloader")
    print("This script will download the COCO 2014 dataset for CLIP training.\n")
    
    # Download and setup dataset
    dataset_path = download_coco_dataset()
    
    # Show statistics
    get_dataset_stats(dataset_path)
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("=" * 80)
    print("\n1. Cache text embeddings (optional but recommended):")
    print("   python cache_text_embeddings.py")
    print("\n2. Start training:")
    print("   python train.py")
    print()
