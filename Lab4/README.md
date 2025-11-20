# ELEC 475 Lab 4: Fine-tuning ResNet50 for CLIP

This project implements a CLIP-style (Contrastive Language-Image Pre-training) model for learning joint image-text embeddings. We fine-tune a ResNet50 image encoder while keeping the CLIP text encoder frozen, training on the MS COCO 2014 dataset with InfoNCE contrastive loss.

## Project Overview

**Goal**: Train a model to learn a shared embedding space where images and their corresponding text captions are close together, enabling:
- Image-to-text retrieval
- Text-to-image retrieval  
- Zero-shot image classification

**Key Components**:
- **Image Encoder**: ResNet50 (pretrained on ImageNet) + projection head → 512-D embeddings
- **Text Encoder**: Frozen CLIP transformer from HuggingFace → 512-D embeddings
- **Loss**: InfoNCE (contrastive loss with temperature scaling)
- **Dataset**: MS COCO 2014 (~118k training images, ~5k validation images)
- **Metrics**: Recall@1, Recall@5, Recall@10 for bidirectional retrieval

## Repository Structure

```
Lab4/
├── config.py                    # Configuration with auto-detection for local/Colab
├── utils.py                     # Training utilities (logger, timer, plotting, checkpointing)
├── download_dataset.py          # Dataset downloader using kagglehub (optional)
├── dataset.py                   # COCO dataset loader with CLIP preprocessing
├── cache_text_embeddings.py     # Pre-encode captions for training speed
├── model.py                     # Baseline CLIP model architecture
├── model_modified.py            # Modified architectures for ablation study
├── loss.py                      # InfoNCE contrastive loss
├── train.py                     # Training pipeline with progress tracking
├── metrics.py                   # Retrieval evaluation metrics
├── test.py                      # Model evaluation script
├── visualize.py                 # Visualization utilities
├── ablation_study.py            # Ablation study framework
├── train.txt                    # Training command reference
├── test.txt                     # Testing command reference
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup Instructions

### Option 1: Local Setup (requires ~19GB disk space + GPU)

1. **Clone repository**:
```bash
git clone https://github.com/Jcub05/475_ML-CV_Labs.git
cd 475_ML-CV_Labs/Lab4
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download COCO 2014 dataset** (choose one method):

   **Method A: Using kagglehub** (automated):
   ```bash
   python download_dataset.py
   ```

   **Method B: Manual download**:
   - Download from [COCO website](https://cocodataset.org/#download):
     - [train2014.zip](http://images.cocodataset.org/zips/train2014.zip) (~13GB)
     - [val2014.zip](http://images.cocodataset.org/zips/val2014.zip) (~6GB)
     - [annotations_trainval2014.zip](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) (~241MB)
   - Extract to `datasets_Lab4/coco_2014/` (outside git repo):
   ```bash
   mkdir -p ../datasets_Lab4/coco_2014
   unzip train2014.zip -d ../datasets_Lab4/coco_2014/
   unzip val2014.zip -d ../datasets_Lab4/coco_2014/
   unzip annotations_trainval2014.zip -d ../datasets_Lab4/coco_2014/
   ```

4. **Cache text embeddings** (run once, ~5 minutes):
```bash
python cache_text_embeddings.py
```

5. **Train baseline model**:
```bash
python train.py
```

### Option 2: Google Colab Setup (recommended for GPU access)

1. **Mount Google Drive and clone repository**:
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/Jcub05/475_ML-CV_Labs.git
%cd 475_ML-CV_Labs/Lab4
```

2. **Install dependencies**:
```python
!pip install -q transformers torch torchvision kagglehub tqdm pillow matplotlib
```

3. **Download dataset to Google Drive** (one-time setup):
   - Option A: Use kagglehub on Colab (downloads to Drive)
   - Option B: Manually upload the 3 zip files to `MyDrive/datasets_Lab4/coco_2014/`

4. **Extract dataset** (if stored as zip files):
```python
!unzip -q /content/drive/MyDrive/datasets_Lab4/coco_2014/train2014.zip -d /content/datasets_Lab4/coco_2014/
!unzip -q /content/drive/MyDrive/datasets_Lab4/coco_2014/val2014.zip -d /content/datasets_Lab4/coco_2014/
!unzip -q /content/drive/MyDrive/datasets_Lab4/coco_2014/annotations_trainval2014.zip -d /content/datasets_Lab4/coco_2014/
```

   **Recommended**: Copy images to local Colab storage for faster training:
```python
!mkdir -p /content/datasets_Lab4/coco_2014
!cp -r /content/drive/MyDrive/datasets_Lab4/coco_2014/train2014 /content/datasets_Lab4/coco_2014/
!cp -r /content/drive/MyDrive/datasets_Lab4/coco_2014/val2014 /content/datasets_Lab4/coco_2014/
!cp -r /content/drive/MyDrive/datasets_Lab4/coco_2014/annotations /content/datasets_Lab4/coco_2014/
```

5. **Cache text embeddings**:
```python
!python cache_text_embeddings.py
```

6. **Train model** (checkpoints auto-save to Google Drive):
```python
!python train.py
```

## Usage

### Training

**Baseline model**:
```bash
python train.py
```

**Custom hyperparameters**:
```bash
python train.py --batch_size 128 --epochs 20 --learning_rate 5e-5
```

**Training details**:
- Default: 15 epochs, batch size 64, learning rate 1e-4
- Optimizer: AdamW with cosine annealing scheduler
- Mixed precision training (AMP) for speed
- Gradient clipping at norm 1.0
- Progress bars show ETA and current metrics
- Best model saved based on validation Recall@5
- Training curves saved to `results/training_curves.png`

### Ablation Study

Run complete ablation study (tests all configurations):
```bash
python ablation_study.py
```

Quick test (baseline + 2 modifications):
```bash
python ablation_study.py --quick
```

Test specific configurations:
```bash
python ablation_study.py --configs baseline batchnorm dropout
```

**Available configurations**:
- `baseline`: Standard ResNet50 + 2-layer projection
- `batchnorm`: Adds BatchNorm to projection head
- `dropout`: Adds Dropout (0.1) to projection head
- `deeper`: 3-layer projection head (2048→1024→512→512)
- `learnable_temp`: Learnable temperature parameter
- `batchnorm_dropout`: Combined BatchNorm + Dropout
- `all_combined`: All modifications together

Results saved to `results/ablation_comparison_table.txt`.

### Testing

**Evaluate trained model**:
```bash
python test.py
```

**Test specific checkpoint**:
```bash
python test.py --checkpoint checkpoints/best_model.pth
```

**Expected baseline performance** (after 15 epochs):
- Image→Text: R@1 ~30-35%, R@5 ~55-65%, R@10 ~70-80%
- Text→Image: R@1 ~25-30%, R@5 ~50-60%, R@10 ~65-75%

### Visualization

**Text-to-image retrieval**:
```bash
python visualize.py --mode text_to_image --queries "a cat sitting on a couch" "a red car on the street"
```

**Image-to-text retrieval**:
```bash
python visualize.py --mode image_to_text --image_path datasets_Lab4/coco_2014/val2014/COCO_val2014_000000000042.jpg
```

**Zero-shot classification**:
```bash
python visualize.py --mode zero_shot --image_path datasets_Lab4/coco_2014/val2014/COCO_val2014_000000000042.jpg --labels "cat" "dog" "car" "person"
```

## Architecture Details

### Image Encoder (Trainable)
```
ResNet50 (pretrained on ImageNet)
  ↓
Flatten: 2048-D
  ↓
Linear: 2048 → 2048
  ↓
GELU activation
  ↓
Linear: 2048 → 512
  ↓
L2 normalization → 512-D embedding
```

### Text Encoder (Frozen)
```
CLIP ViT-B/32 from HuggingFace
  ↓
Frozen transformer
  ↓
Pooler output: 512-D
  ↓
L2 normalization → 512-D embedding
```

### InfoNCE Loss
```python
# Compute similarity matrix
logits = image_embeddings @ text_embeddings.T / temperature

# Symmetric loss
loss_i2t = CrossEntropy(logits, targets)  # Image → Text
loss_t2i = CrossEntropy(logits.T, targets)  # Text → Image
loss = (loss_i2t + loss_t2i) / 2
```

Temperature = 0.07 (controls prediction sharpness)

## Modifications (Ablation Study)

1. **BatchNorm**: Adds batch normalization after each linear layer in projection head
   - Stabilizes training and improves generalization

2. **Dropout**: Adds dropout (p=0.1) after activations in projection head
   - Reduces overfitting on training set

3. **Deeper Projection**: 3-layer projection head instead of 2-layer
   - Increases model capacity for better feature transformation

4. **Learnable Temperature**: Makes temperature a learnable parameter
   - Allows model to adaptively adjust prediction confidence

## Hardware Requirements

- **GPU**: NVIDIA GPU with 10GB+ VRAM (RTX 3080, V100, A100, etc.)
- **RAM**: 16GB+ system RAM
- **Storage**: 19GB for dataset + 5GB for checkpoints/cache
- **Training time**: ~2-4 hours per epoch on RTX 3080 / A100

**Batch size guidelines**:
- Batch size 32: ~6GB GPU memory
- Batch size 64: ~10GB GPU memory
- Batch size 128: ~16GB GPU memory

## File Locations

### Local Setup
- Dataset: `datasets_Lab4/coco_2014/`
- Checkpoints: `checkpoints/`
- Results: `results/`
- Logs: `checkpoints/training.log`

### Google Colab
- Dataset: `/content/datasets_Lab4/coco_2014/` (copied from Drive)
- Checkpoints: `/content/drive/MyDrive/datasets_Lab4/Lab4_checkpoints/`
- Results: `/content/drive/MyDrive/datasets_Lab4/Lab4_results/`
- Logs: `/content/drive/MyDrive/datasets_Lab4/Lab4_checkpoints/training.log`

## Troubleshooting

**Out of GPU memory**:
- Reduce batch size: `python train.py --batch_size 32`
- Clear cache: `torch.cuda.empty_cache()`

**Slow training**:
- Ensure text embeddings are cached (run `cache_text_embeddings.py`)
- Copy dataset to local Colab storage (not reading from Drive)
- Use mixed precision training (enabled by default)

**Dataset not found**:
- Verify directory structure matches `datasets_Lab4/coco_2014/train2014/`, etc.
- Check `config.py` for correct paths

**Colab session timeout**:
- Checkpoints auto-save to Google Drive every epoch
- Resume training by re-running `train.py` (loads latest checkpoint)

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models From Natural Language Supervision
- [COCO Dataset](https://cocodataset.org/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Lab Deliverables

1. ✅ Code implementation (all `.py` files)
2. ✅ `train.txt` - Training commands
3. ✅ `test.txt` - Testing commands
4. ⏳ Trained model weights (`best_model.pth`)
5. ⏳ Training curves and retrieval metrics plots
6. ⏳ Ablation study comparison table
7. ⏳ Lab report with analysis and results

## License

This project is for educational purposes as part of ELEC 475 coursework at Queen's University.

## Author

Jacob Cubin (jcube)
Queen's University - ELEC 475 - Fall 2024
