import os
from typing import Tuple, Optional

from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dataset import OxfordPetNosesDataset
from transforms_geom import maybe_hflip


def build_transforms(normalize_imagenet: bool = False, color_jitter: bool = False):
    transforms_list = []
    
    # Add color jitter if requested (doesn't affect coordinates)
    if color_jitter:
        transforms_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    
    # Convert to tensor
    transforms_list.append(T.ToTensor())
    
    # Normalize if requested
    if normalize_imagenet:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transforms_list.append(T.Normalize(mean=mean, std=std))
    
    return T.Compose(transforms_list)


def get_datasets(root: str,
                 augment: bool = False,
                 normalize_imagenet: bool = False,
                 output_size: int = 227):
    images_dir = os.path.join(root, "oxford-iiit-pet-noses", "images-original", "images")
    train_labels = os.path.join(root, "oxford-iiit-pet-noses", "train_noses.txt")
    test_labels = os.path.join(root, "oxford-iiit-pet-noses", "test_noses.txt")

    # Enable color jitter when augmentation is on
    to_tensor_norm = build_transforms(normalize_imagenet=normalize_imagenet, color_jitter=augment)

    # Enable horizontal flip when augmentation is on
    geom = maybe_hflip(0.5) if augment else None

    train_ds = OxfordPetNosesDataset(
        images_dir=images_dir,
        labels_file=train_labels,
        output_size=output_size,
        geom_transform=geom,
        to_tensor_normalize=to_tensor_norm,
    )

    test_ds = OxfordPetNosesDataset(
        images_dir=images_dir,
        labels_file=test_labels,
        output_size=output_size,
        geom_transform=None,  # no aug in test
        to_tensor_normalize=to_tensor_norm,
    )

    return train_ds, test_ds


def get_dataloaders(root: str,
                    batch_size: int = 32,
                    num_workers: int = 2,
                    augment: bool = False,
                    normalize_imagenet: bool = False,
                    output_size: int = 227):
    train_ds, test_ds = get_datasets(
        root=root,
        augment=augment,
        normalize_imagenet=normalize_imagenet,
        output_size=output_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


