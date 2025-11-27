"""
Evaluation script for computing Recall@K metrics after training.
Runs separately from training to avoid memory issues.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from config import get_config
from dataset import create_dataloaders
from model import CLIPFineTuneModel
from metrics import compute_retrieval_metrics
from utils import Logger, plot_recall_metrics


@torch.no_grad()
def evaluate_model(model, dataloader, device, config, logger):
    """
    Evaluate model and compute Recall@K metrics.
    
    Args:
        model: CLIP model
        dataloader: Validation dataloader
        device: Device
        config: Configuration
        logger: Logger
        
    Returns:
        Dictionary with Recall@K metrics
    """
    model.eval()
    
    # Collect all embeddings
    all_image_embeds = []
    all_text_embeds = []
    
    logger.log("Collecting embeddings...")
    pbar = tqdm(dataloader, desc="Evaluation")
    
    for batch in pbar:
        images = batch['image'].to(device)
        text_embeddings = batch['text_embedding'].to(device)
        
        # Forward pass
        image_embeds, text_embeds = model(images, text_embeddings=text_embeddings)
        
        # Store embeddings (move to CPU to save GPU memory)
        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())
        
        # Clear GPU cache periodically
        if len(all_image_embeds) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    logger.log("Computing Recall@K metrics...")
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # Compute Recall@K metrics (move back to GPU for fast computation)
    recall_metrics = compute_retrieval_metrics(
        all_image_embeds.to(device),
        all_text_embeds.to(device),
        k_values=config.recall_k_values
    )
    
    # Clear memory
    del all_image_embeds, all_text_embeds
    torch.cuda.empty_cache()
    
    return recall_metrics


def main(args):
    """Main evaluation function."""
    # Get configuration
    config = get_config()
    device = config.device
    
    # Logger
    log_file = config.results_path / "evaluation_log.txt"
    logger = Logger(log_file=log_file, verbose=True)
    
    logger.log("=" * 80)
    logger.log("CLIP Model Evaluation - Recall@K Computation")
    logger.log("=" * 80)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.log(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    logger.log("Creating model...")
    model = CLIPFineTuneModel(
        embed_dim=config.embed_dim,
        pretrained_resnet=config.pretrained_resnet,
        clip_model_name=config.clip_model_name,
        freeze_text_encoder=True
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.log(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create validation dataloader
    logger.log("Creating validation dataloader...")
    _, val_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        use_cached_embeddings=config.use_cached_embeddings,
        use_subset=config.use_subset,
        subset_size=config.subset_size
    )
    
    # Evaluate
    logger.log("\n" + "=" * 80)
    logger.log("Starting Evaluation")
    logger.log("=" * 80 + "\n")
    
    metrics = evaluate_model(model, val_loader, device, config, logger)
    
    # Log results
    logger.log("\n" + "=" * 80)
    logger.log("Evaluation Results")
    logger.log("=" * 80)
    logger.log(f"Image→Text Retrieval:")
    logger.log(f"  R@1:  {metrics['img2txt_r1']:.2f}%")
    logger.log(f"  R@5:  {metrics['img2txt_r5']:.2f}%")
    logger.log(f"  R@10: {metrics['img2txt_r10']:.2f}%")
    logger.log(f"\nText→Image Retrieval:")
    logger.log(f"  R@1:  {metrics['txt2img_r1']:.2f}%")
    logger.log(f"  R@5:  {metrics['txt2img_r5']:.2f}%")
    logger.log(f"  R@10: {metrics['txt2img_r10']:.2f}%")
    logger.log(f"\nAverage Recall: {metrics['avg_recall']:.2f}%")
    logger.log("=" * 80)
    
    # Save metrics
    metrics_file = config.results_path / "recall_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.log(f"\nMetrics saved: {metrics_file}")
    
    # Plot metrics
    plot_path = config.results_path / "recall_metrics.png"
    plot_recall_metrics(metrics, plot_path)
    logger.log(f"Plot saved: {plot_path}")
    
    logger.log("\nEvaluation complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate CLIP model Recall@K")
    
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
