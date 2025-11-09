"""
Test script for pretrained FCN-ResNet50 on PASCAL VOC 2012 dataset
ELEC 475 Lab 3 - Step 1
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path


def calculate_miou(pred, target, num_classes=21):
    """
    Calculate mean Intersection over Union (mIoU) metric
    
    Args:
        pred: predicted segmentation mask (H, W)
        target: ground truth segmentation mask (H, W)
        num_classes: number of classes (21 for VOC)
    
    Returns:
        mIoU value
    """
    ious = []
    # Ignore index 255 (boundary/void class in VOC)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            # If this class doesn't appear in target, skip it
            continue
        
        iou = intersection / union
        ious.append(iou)
    
    return np.mean(ious) if len(ious) > 0 else 0.0


class VOCNormalize:
    """Custom normalization for VOC dataset"""
    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, img, target):
        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return img, target


def collate_fn(batch):
    """Custom collate function to handle variable-sized images"""
    images, targets = zip(*batch)
    return list(images), list(targets)


def evaluate_model(model, dataloader, device, num_classes=21):
    """
    Evaluate the model on the dataset
    
    Args:
        model: segmentation model
        dataloader: data loader
        device: device to run on
        num_classes: number of classes
    
    Returns:
        mean IoU across all samples
    """
    model.eval()
    all_ious = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            for img, target in zip(images, targets):
                # Add batch dimension
                img = img.unsqueeze(0).to(device)
                target = target.to(device)
                
                # Forward pass
                output = model(img)['out']
                
                # Get predictions
                pred = output.squeeze(0).argmax(0)
                
                # Calculate IoU for this sample
                iou = calculate_miou(pred, target, num_classes)
                all_ious.append(iou)
    
    mean_iou = np.mean(all_ious)
    return mean_iou


def main():
    """Main function to test pretrained FCN-ResNet50"""
    
    print("=" * 60)
    print("Testing Pretrained FCN-ResNet50 on PASCAL VOC 2012")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load pretrained FCN-ResNet50
    print("\nLoading pretrained FCN-ResNet50...")
    from torchvision.models.segmentation import FCN_ResNet50_Weights
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.fcn_resnet50(weights=weights)
    model = model.to(device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Download and prepare PASCAL VOC 2012 dataset
    print("\nPreparing PASCAL VOC 2012 dataset...")
    
    transform = VOCNormalize()
    
    # Try multiple possible dataset locations
    possible_roots = [
        './data',  # Local data folder
        str(Path.home() / '.cache' / 'kagglehub' / 'datasets' / 'huanghanchina' / 'pascal-voc-2012' / 'versions' / '1'),  # Kaggle cache
    ]
    
    # Also check if we saved the path
    dataset_path_file = Path('dataset_path.txt')
    if dataset_path_file.exists():
        with open(dataset_path_file, 'r') as f:
            saved_path = Path(f.read().strip())
            # The saved path is to VOC2012, we need the parent's parent for the root
            possible_roots.insert(0, str(saved_path.parent.parent))
    
    val_dataset = None
    dataset_root = None
    
    for root in possible_roots:
        try:
            print(f"  Trying: {root}")
            val_dataset = VOCSegmentation(
                root=root,
                year='2012',
                image_set='val',
                download=False,
                transforms=transform
            )
            dataset_root = root
            print(f"  ✓ Dataset found!")
            break
        except Exception as e:
            print(f"    Not found here")
            continue
    
    if val_dataset is None:
        print(f"\nDataset not found in any location.")
        print("\nTo download the dataset, run:")
        print("  python download_voc_kaggle.py")
        print("\nThis will download the dataset using Kaggle Hub.")
        print("You'll need a Kaggle account and API token.")
        return
    
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        collate_fn=collate_fn
    )
    
    # Evaluate the model
    print("\nEvaluating model on validation set...")
    print("This may take a few minutes...\n")
    
    miou = evaluate_model(model, val_loader, device, num_classes=21)
    
    print("\n" + "=" * 60)
    print(f"Results:")
    print(f"  Mean IoU (mIoU): {miou:.4f} ({miou*100:.2f}%)")
    print(f"  Number of parameters: {num_params:,}")
    print("=" * 60)
    
    # Save results
    results = {
        'model': 'FCN-ResNet50 (pretrained)',
        'mIoU': miou,
        'num_parameters': num_params
    }
    
    torch.save(results, 'pretrained_fcn_results.pt')
    print("\nResults saved to 'pretrained_fcn_results.pt'")


if __name__ == '__main__':
    main()
