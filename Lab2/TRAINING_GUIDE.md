# Training Guide for Pet Nose Localization

## Overview
This project includes a unified training pipeline that supports three model architectures:
- **SnoutNet**: Custom CNN architecture
- **SnoutNet-A**: AlexNet backbone with regression head
- **SnoutNet-V**: VGG16 backbone with regression head

## Unified Scripts

### `train.py`
Main training script that accepts `--model_type` to train any model.

**Key arguments:**
- `--model_type`: Choose from `snoutnet`, `alexnet`, or `vgg16`
- `--augment`: Enable data augmentation (horizontal flip + color jitter)
- `--normalize_imagenet`: Use ImageNet normalization (required for pretrained models)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001 for SnoutNet, 0.0001 for pretrained)
- `--cuda`: Use GPU if available

### `test.py`
Testing script that calculates localization accuracy statistics.

**Outputs:**
- Min, mean, median, max, std of Euclidean distance errors
- Error distribution histogram
- Cumulative distribution plot

### `visualize.py`
Visualization script that shows predictions vs ground truth.

**Outputs:**
- Visual comparison of predicted vs actual nose locations
- Error distance for each sample

## Training Workflow

### For Google Colab (with dataset in `/content/`)
```bash
# SnoutNet without augmentation
python train.py --model_type snoutnet --epochs 50 --batch_size 32 --learning_rate 0.001 --output_dir ./SnoutNet/snoutnet_training --data_root /content --num_workers 0 --cuda

# SnoutNet with augmentation
python train.py --model_type snoutnet --epochs 50 --batch_size 32 --learning_rate 0.001 --output_dir ./SnoutNet/snoutnet_training_augmented --data_root /content --num_workers 0 --augment --cuda

# AlexNet without augmentation (NOTE: lower learning rate + ImageNet normalization)
python train.py --model_type alexnet --epochs 50 --batch_size 32 --learning_rate 0.0001 --output_dir ./SnoutNet-A/alexnet_training --data_root /content --num_workers 0 --normalize_imagenet --cuda

# AlexNet with augmentation
python train.py --model_type alexnet --epochs 50 --batch_size 32 --learning_rate 0.0001 --output_dir ./SnoutNet-A/alexnet_training_augmented --data_root /content --num_workers 0 --normalize_imagenet --augment --cuda

# VGG16 without augmentation (NOTE: lower learning rate + ImageNet normalization)
python train.py --model_type vgg16 --epochs 50 --batch_size 32 --learning_rate 0.0001 --output_dir ./SnoutNet-V/vgg16_training --data_root /content --num_workers 0 --normalize_imagenet --cuda

# VGG16 with augmentation
python train.py --model_type vgg16 --epochs 50 --batch_size 32 --learning_rate 0.0001 --output_dir ./SnoutNet-V/vgg16_training_augmented --data_root /content --num_workers 0 --normalize_imagenet --augment --cuda
```

### For Local Training (Windows)
```bash
# SnoutNet without augmentation
python train.py --model_type snoutnet --epochs 50 --batch_size 32 --learning_rate 0.001 --output_dir ./SnoutNet/snoutnet_training --cuda

# SnoutNet with augmentation
python train.py --model_type snoutnet --epochs 50 --batch_size 32 --learning_rate 0.001 --output_dir ./SnoutNet/snoutnet_training_augmented --augment --cuda

# AlexNet without augmentation
python train.py --model_type alexnet --epochs 50 --batch_size 32 --learning_rate 0.0001 --output_dir ./SnoutNet-A/alexnet_training --normalize_imagenet --cuda

# AlexNet with augmentation
python train.py --model_type alexnet --epochs 50 --batch_size 32 --learning_rate 0.0001 --output_dir ./SnoutNet-A/alexnet_training_augmented --normalize_imagenet --augment --cuda

# VGG16 without augmentation
python train.py --model_type vgg16 --epochs 50 --batch_size 32 --learning_rate 0.0001 --output_dir ./SnoutNet-V/vgg16_training --normalize_imagenet --cuda

# VGG16 with augmentation
python train.py --model_type vgg16 --epochs 50 --batch_size 32 --learning_rate 0.0001 --output_dir ./SnoutNet-V/vgg16_training_augmented --normalize_imagenet --augment --cuda
```

## Testing

After training, test each model:

```bash
# Test SnoutNet
python test.py --model_path ./SnoutNet/snoutnet_training/best_snoutnet.pth --output_dir ./SnoutNet/test_results --cuda

# Test AlexNet
python test.py --model_path ./SnoutNet-A/alexnet_training/best_alexnet.pth --output_dir ./SnoutNet-A/test_results --normalize_imagenet --cuda

# Test VGG16
python test.py --model_path ./SnoutNet-V/vgg16_training/best_vgg16.pth --output_dir ./SnoutNet-V/test_results --normalize_imagenet --cuda
```

## Visualization

Generate visualizations:

```bash
# Visualize SnoutNet
python visualize.py --model_path ./SnoutNet/snoutnet_training/best_snoutnet.pth --output_dir ./SnoutNet/visualizations --num_samples 8 --cuda

# Visualize AlexNet
python visualize.py --model_path ./SnoutNet-A/alexnet_training/best_alexnet.pth --output_dir ./SnoutNet-A/visualizations --num_samples 8 --normalize_imagenet --cuda

# Visualize VGG16
python visualize.py --model_path ./SnoutNet-V/vgg16_training/best_vgg16.pth --output_dir ./SnoutNet-V/visualizations --num_samples 8 --normalize_imagenet --cuda
```

## Important Notes

### Data Augmentation
When `--augment` is enabled, two augmentations are applied:
1. **Horizontal Flip** (p=0.5): Randomly flips image horizontally with automatic coordinate adjustment
2. **Color Jitter**: Randomly adjusts brightness (±20%), contrast (±20%), saturation (±20%), and hue (±10%)

### ImageNet Normalization
- **Required** for AlexNet and VGG16 (pretrained models)
- **Not used** for SnoutNet (trained from scratch)
- Uses mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Learning Rates
- **SnoutNet**: 0.001 (trained from scratch)
- **AlexNet/VGG16**: 0.0001 (fine-tuning pretrained models, all layers trainable)

### Model Checkpointing
Training automatically saves:
- `best_{model_type}.pth`: Model with lowest validation loss
- `final_{model_type}.pth`: Model after final epoch
- `training_curves.png`: Loss and distance plots

## Folder Structure

```
Lab 2/
├── train.py                    # Unified training script
├── test.py                     # Unified testing script
├── visualize.py                # Unified visualization script
├── model.py                    # SnoutNet architecture
├── dataset.py                  # Custom dataset class
├── datamodule.py               # DataLoader factory
├── transforms_geom.py          # Geometric transformations
├── SnoutNet/
│   ├── SnoutNet.txt           # SnoutNet commands
│   ├── model_alexnet.py       # AlexNet model
│   ├── snoutnet_training/     # Training outputs (no aug)
│   ├── snoutnet_training_augmented/  # Training outputs (aug)
│   ├── test_results/          # Testing outputs
│   └── visualizations/        # Visualization outputs
├── SnoutNet-A/
│   ├── SnoutNet-A.txt         # AlexNet commands
│   ├── model_alexnet.py       # AlexNet model
│   ├── alexnet_training/      # Training outputs (no aug)
│   ├── alexnet_training_augmented/  # Training outputs (aug)
│   ├── test_results/          # Testing outputs
│   └── visualizations/        # Visualization outputs
├── SnoutNet-V/
│   ├── SnoutNet-V.txt         # VGG16 commands
│   ├── model_vgg16.py         # VGG16 model
│   ├── vgg16_training/        # Training outputs (no aug)
│   ├── vgg16_training_augmented/  # Training outputs (aug)
│   ├── test_results/          # Testing outputs
│   └── visualizations/        # Visualization outputs
└── SnoutNet-Ensemble/
    └── (ensemble implementation)
```

## Training Time Estimates
- **SnoutNet**: ~10-15 minutes per epoch on GPU
- **AlexNet**: ~15-20 minutes per epoch on GPU
- **VGG16**: ~20-30 minutes per epoch on GPU

Testing is much faster: ~1-2 minutes per model.

## Next Steps
1. Train all 6 model variants (3 architectures × 2 augmentation settings)
2. Test each trained model
3. Generate visualizations
4. Implement ensemble model (Step 7)
5. Compare results in report




