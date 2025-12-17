# Lab 3: Semantic Segmentation with Model Compression

## Objective
Develop a parameter-efficient semantic segmentation model for PASCAL VOC 2012 dataset (21 classes). Models were evaluated using the formula: **Score = 4 × mIoU / (1 + params_M)**, prioritizing both accuracy and parameter efficiency.

**Result: 2nd place in class competition**

## Architecture
- **Backbone**: Truncated MobileNetV3-Small (10 stages instead of 13)
- **Decoder**: Simplified ASPP module with 2 dilation rates (6, 12)
- **Parameters**: 0.48M (475K)

## Models Trained

### 1. Baseline (No Knowledge Distillation)
- **mIoU**: 29.99%
- **Score**: 0.8133
- **Training Time**: 0.48h

### 2. Response-Based Knowledge Distillation
- **mIoU**: 39.41%
- **Score**: 1.0688 ⭐ (Best)
- **Training Time**: 0.46h
- Uses soft targets from teacher model with temperature scaling

### 3. Feature-Based Knowledge Distillation
- **mIoU**: 38.97%
- **Score**: 1.0566
- **Training Time**: 0.59h
- Matches intermediate feature representations between student and teacher

## Key Techniques
- Backbone truncation to minimize parameters
- ASPP (Atrous Spatial Pyramid Pooling) for multi-scale context
- Knowledge distillation from larger teacher model
- Single-stage decoder for efficiency

## Files
- `model_ultracompact.py` - Main model architecture
- `train.py` - Training script with KD support
- `test.py` - Evaluation script
- `Results/` - Training results and comparison tables
