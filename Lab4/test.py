"""
Testing and evaluation script for trained CLIP model.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from config import get_config
from dataset import create_dataloaders
from model import CLIPFineTuneModel
from metrics import compute_retrieval_metrics, compute_mean_rank
from loss import InfoNCELossWithMetrics
from utils import Logger, plot_recall_metrics, load_checkpoint


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device: str,
    logger: Logger,
    compute_loss: bool = True
):
    """
    Evaluate model on a dataset.
    
    Args:
        model: CLIP model
        dataloader: DataLoader
        device: Device
        logger: Logger
        compute_loss: Whether to compute loss
        
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    # Collect embeddings
    all_image_embeds = []
    all_text_embeds = []
    total_loss = 0.0
    num_batches = 0
    
    criterion = InfoNCELossWithMetrics(temperature=0.07) if compute_loss else None
    
    logger.log("Evaluating model...")
    for batch in tqdm(dataloader, desc="Evaluation"):
        images = batch['image'].to(device)
        text_embeddings = batch['text_embedding'].to(device)
        
        # Forward pass
        image_embeds, text_embeds = model(images, text_embeddings=text_embeddings)
        
        # Compute loss if needed
        if compute_loss and criterion is not None:
            loss, _ = criterion(image_embeds, text_embeds, return_metrics=True)
            total_loss += loss.item()
            num_batches += 1
        
        # Store embeddings
        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())
    
    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    logger.log(f"Collected {len(all_image_embeds)} image-text pairs")
    
    # Compute retrieval metrics
    logger.log("Computing Recall@K metrics...")
    recall_metrics = compute_retrieval_metrics(
        all_image_embeds,
        all_text_embeds,
        k_values=[1, 5, 10]
    )
    
    # Compute mean rank
    logger.log("Computing mean ranks...")
    from metrics import compute_similarity_matrix
    similarity = compute_similarity_matrix(all_image_embeds, all_text_embeds)
    rank_metrics = compute_mean_rank(similarity)
    
    # Combine metrics
    metrics = {
        **recall_metrics,
        **rank_metrics
    }
    
    if compute_loss:
        metrics['loss'] = total_loss / num_batches
    
    return metrics, all_image_embeds, all_text_embeds


def print_metrics(metrics: dict, logger: Logger):
    """Print metrics in a nice format."""
    logger.log("\n" + "=" * 80)
    logger.log("EVALUATION RESULTS")
    logger.log("=" * 80)
    
    if 'loss' in metrics:
        logger.log(f"\nLoss: {metrics['loss']:.4f}")
    
    logger.log("\nImage → Text Retrieval:")
    logger.log(f"  Recall@1:  {metrics['img2txt_r1']:.2f}%")
    logger.log(f"  Recall@5:  {metrics['img2txt_r5']:.2f}%")
    logger.log(f"  Recall@10: {metrics['img2txt_r10']:.2f}%")
    logger.log(f"  Mean Rank: {metrics['img2txt_mean_rank']:.2f}")
    
    logger.log("\nText → Image Retrieval:")
    logger.log(f"  Recall@1:  {metrics['txt2img_r1']:.2f}%")
    logger.log(f"  Recall@5:  {metrics['txt2img_r5']:.2f}%")
    logger.log(f"  Recall@10: {metrics['txt2img_r10']:.2f}%")
    logger.log(f"  Mean Rank: {metrics['txt2img_mean_rank']:.2f}")
    
    logger.log("\nAverage Performance:")
    logger.log(f"  Avg Image→Text: {metrics['avg_img2txt_recall']:.2f}%")
    logger.log(f"  Avg Text→Image: {metrics['avg_txt2img_recall']:.2f}%")
    logger.log(f"  Overall Avg:    {metrics['avg_recall']:.2f}%")
    logger.log("=" * 80 + "\n")


def main(args):
    """Main testing function."""
    # Get config
    config_kwargs = {}
    if args.data_root:
        config_kwargs['data_root'] = args.data_root
    if args.batch_size:
        config_kwargs['batch_size'] = args.batch_size
    
    config = get_config(**config_kwargs)
    
    # Validate dataset
    config.validate_paths()
    
    # Setup logger
    log_file = config.results_path / "test_log.txt"
    logger = Logger(log_file=log_file, verbose=True)
    
    logger.log("=" * 80)
    logger.log("CLIP Model Testing")
    logger.log("=" * 80)
    logger.log(f"Checkpoint: {args.checkpoint}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}\n")
    
    # Create model
    logger.log("Creating model...")
    model = CLIPFineTuneModel(
        embed_dim=config.embed_dim,
        pretrained_resnet=config.pretrained_resnet,
        clip_model_name=config.clip_model_name,
        freeze_text_encoder=True
    ).to(device)
    
    # Load checkpoint
    logger.log(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        # Try looking in checkpoint directory
        checkpoint_path = config.checkpoint_path / args.checkpoint
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint_info = load_checkpoint(checkpoint_path, model, device=device)
    logger.log(f"✓ Loaded checkpoint from epoch {checkpoint_info['epoch']}")
    
    # Create dataloader
    logger.log("\nCreating dataloader...")
    
    if args.split == 'val':
        _, dataloader = create_dataloaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            use_cached_embeddings=True,
            use_subset=False
        )
    else:
        dataloader, _ = create_dataloaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            use_cached_embeddings=True,
            use_subset=False
        )
    
    logger.log(f"Evaluating on {args.split} split")
    logger.log(f"Number of batches: {len(dataloader)}\n")
    
    # Evaluate
    metrics, image_embeds, text_embeds = evaluate_model(
        model, dataloader, device, logger, compute_loss=True
    )
    
    # Print metrics
    print_metrics(metrics, logger)
    
    # Save metrics
    import json
    metrics_file = config.results_path / f"test_metrics_{args.split}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.log(f"Metrics saved to: {metrics_file}")
    
    # Generate plots
    if args.save_plots:
        logger.log("\nGenerating plots...")
        plot_path = config.results_path / f"test_metrics_{args.split}.png"
        plot_recall_metrics(metrics, plot_path, title=f"Retrieval Performance ({args.split.upper()})")
        logger.log(f"Plot saved to: {plot_path}")
    
    # Save embeddings if requested
    if args.save_embeddings:
        logger.log("\nSaving embeddings...")
        embeddings_file = config.results_path / f"embeddings_{args.split}.pt"
        torch.save({
            'image_embeds': image_embeds,
            'text_embeds': text_embeds
        }, embeddings_file)
        logger.log(f"Embeddings saved to: {embeddings_file}")
    
    logger.log("\n✓ Testing complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test CLIP model")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint or checkpoint name')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory of COCO dataset')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save metric plots')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Save computed embeddings')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
