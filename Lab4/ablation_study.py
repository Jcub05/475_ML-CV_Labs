"""
Ablation Study Framework for CLIP Fine-tuning
Systematically evaluates baseline and modified architectures.
"""

import os
import json
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

from config import Config
from dataset import create_dataloaders
from model import create_clip_model
from model_modified import create_modified_model, MODEL_CONFIGS
from metrics import compute_retrieval_metrics
from utils import Logger, load_checkpoint


def evaluate_model(model, val_loader, device, config):
    """
    Evaluate a model on validation set.
    
    Args:
        model: CLIP model to evaluate
        val_loader: Validation data loader
        device: torch device
        config: Config object
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_image_embeddings = []
    all_text_embeddings = []
    
    print("Computing embeddings...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device)
            text_embeddings = batch['text_embedding'].to(device)
            
            # Forward pass
            image_embeddings, text_embeddings, _ = model(images, text_embeddings)
            
            all_image_embeddings.append(image_embeddings.cpu())
            all_text_embeddings.append(text_embeddings.cpu())
    
    # Concatenate all embeddings
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    print("Computing retrieval metrics...")
    metrics = compute_retrieval_metrics(
        all_image_embeddings,
        all_text_embeddings,
        k_values=config.recall_k_values
    )
    
    return metrics


def train_and_evaluate_config(config_name, model_config, config, device):
    """
    Train and evaluate a single model configuration.
    
    Args:
        config_name: Name of the configuration
        model_config: Dictionary with model parameters
        config: Config object
        device: torch device
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Training configuration: {config_name}")
    print(f"{'='*80}")
    print(f"Parameters: {model_config}")
    
    # Create output directory for this configuration
    config_dir = os.path.join(config.checkpoint_dir, f"ablation_{config_name}")
    os.makedirs(config_dir, exist_ok=True)
    
    # Initialize logger
    logger = Logger(os.path.join(config_dir, 'training.log'))
    logger.log(f"Configuration: {config_name}")
    logger.log(f"Parameters: {json.dumps(model_config, indent=2)}")
    
    # Load text encoder and tokenizer
    print("Loading text encoder and tokenizer...")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    
    # Create model based on configuration
    if config_name == 'baseline':
        # Use original model
        from model import create_clip_model
        model = create_clip_model(text_encoder, tokenizer, config.embed_dim)
    else:
        # Use modified model
        model = create_modified_model(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            embed_dim=config.embed_dim,
            **model_config
        )
    
    model = model.to(device)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    # Train the model (import and use train_model from train.py)
    print("Starting training...")
    from train import train_model
    
    # Override checkpoint directory temporarily
    original_checkpoint_dir = config.checkpoint_dir
    config.checkpoint_dir = config_dir
    
    best_metrics = train_model(model, train_loader, val_loader, config, device)
    
    # Restore original checkpoint directory
    config.checkpoint_dir = original_checkpoint_dir
    
    # Load best checkpoint
    best_checkpoint_path = os.path.join(config_dir, 'best_model.pth')
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best checkpoint from {best_checkpoint_path}")
        model, _, _, _ = load_checkpoint(best_checkpoint_path, model, device=device)
    
    # Final evaluation
    print("Running final evaluation...")
    final_metrics = evaluate_model(model, val_loader, device, config)
    
    # Save results
    results = {
        'config_name': config_name,
        'model_config': model_config,
        'best_val_metrics': best_metrics,
        'final_metrics': final_metrics
    }
    
    results_path = os.path.join(config_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.log(f"\nFinal Results:")
    logger.log(f"Image->Text Retrieval:")
    logger.log(f"  Recall@1:  {final_metrics['image_to_text']['recall@1']:.4f}")
    logger.log(f"  Recall@5:  {final_metrics['image_to_text']['recall@5']:.4f}")
    logger.log(f"  Recall@10: {final_metrics['image_to_text']['recall@10']:.4f}")
    logger.log(f"\nText->Image Retrieval:")
    logger.log(f"  Recall@1:  {final_metrics['text_to_image']['recall@1']:.4f}")
    logger.log(f"  Recall@5:  {final_metrics['text_to_image']['recall@5']:.4f}")
    logger.log(f"  Recall@10: {final_metrics['text_to_image']['recall@10']:.4f}")
    
    return results


def run_ablation_study(configs_to_test=None):
    """
    Run complete ablation study across multiple configurations.
    
    Args:
        configs_to_test: List of config names to test. If None, tests all.
    """
    config = Config()
    device = torch.device(config.device)
    
    # Determine which configurations to test
    if configs_to_test is None:
        configs_to_test = list(MODEL_CONFIGS.keys())
    
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY")
    print(f"{'='*80}")
    print(f"Configurations to test: {configs_to_test}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Run experiments for each configuration
    all_results = {}
    for config_name in configs_to_test:
        if config_name not in MODEL_CONFIGS:
            print(f"Warning: Unknown configuration '{config_name}', skipping...")
            continue
        
        model_config = MODEL_CONFIGS[config_name]
        results = train_and_evaluate_config(config_name, model_config, config, device)
        all_results[config_name] = results
    
    # Save combined results
    combined_results_path = os.path.join(config.results_dir, 'ablation_study_results.json')
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate comparison table
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS - COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    # Create comparison table
    print(f"{'Configuration':<20} {'I->T R@1':<10} {'I->T R@5':<10} {'I->T R@10':<11} {'T->I R@1':<10} {'T->I R@5':<10} {'T->I R@10':<11}")
    print("-" * 100)
    
    for config_name, results in all_results.items():
        metrics = results['final_metrics']
        i2t = metrics['image_to_text']
        t2i = metrics['text_to_image']
        
        print(f"{config_name:<20} "
              f"{i2t['recall@1']:<10.4f} "
              f"{i2t['recall@5']:<10.4f} "
              f"{i2t['recall@10']:<11.4f} "
              f"{t2i['recall@1']:<10.4f} "
              f"{t2i['recall@5']:<10.4f} "
              f"{t2i['recall@10']:<11.4f}")
    
    # Save table to file
    table_path = os.path.join(config.results_dir, 'ablation_comparison_table.txt')
    with open(table_path, 'w') as f:
        f.write("ABLATION STUDY RESULTS - COMPARISON TABLE\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Configuration':<20} {'I->T R@1':<10} {'I->T R@5':<10} {'I->T R@10':<11} {'T->I R@1':<10} {'T->I R@5':<10} {'T->I R@10':<11}\n")
        f.write("-" * 100 + "\n")
        
        for config_name, results in all_results.items():
            metrics = results['final_metrics']
            i2t = metrics['image_to_text']
            t2i = metrics['text_to_image']
            
            f.write(f"{config_name:<20} "
                   f"{i2t['recall@1']:<10.4f} "
                   f"{i2t['recall@5']:<10.4f} "
                   f"{i2t['recall@10']:<11.4f} "
                   f"{t2i['recall@1']:<10.4f} "
                   f"{t2i['recall@5']:<10.4f} "
                   f"{t2i['recall@10']:<11.4f}\n")
    
    print(f"\nResults saved to:")
    print(f"  - {combined_results_path}")
    print(f"  - {table_path}")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CLIP ablation study')
    parser.add_argument('--configs', nargs='+', default=None,
                       help='Specific configurations to test (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with baseline, batchnorm, and dropout only')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick ablation: baseline + two modifications
        configs_to_test = ['baseline', 'batchnorm', 'dropout']
    else:
        configs_to_test = args.configs
    
    run_ablation_study(configs_to_test)
