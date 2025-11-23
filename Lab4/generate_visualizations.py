"""
Generate all required visualizations for Lab 4 report.
Run this after training to create qualitative examples.
"""

import os
import torch
import json
from pathlib import Path
from transformers import CLIPProcessor
from PIL import Image
import random

from config import Config
from model import create_clip_model
from dataset import get_clip_transforms
from visualize import (
    visualize_text_to_image_retrieval,
    visualize_image_to_text_retrieval,
    zero_shot_classification,
    create_retrieval_grid
)
from utils import load_checkpoint


def load_model_and_data(checkpoint_path, config, device):
    """Load trained model and prepare data."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model
    from transformers import CLIPTokenizer, CLIPTextModel
    text_encoder = CLIPTextModel.from_pretrained(config.clip_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(config.clip_model_name)
    
    model = create_clip_model(text_encoder, tokenizer, config.embed_dim)
    model = model.to(device)
    
    # Load checkpoint
    model, _, _, _ = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def load_validation_data(config):
    """Load validation image paths and pre-computed embeddings."""
    from dataset import COCOImageCaptionDataset
    
    # Load dataset to get image paths and captions
    val_dataset = COCOImageCaptionDataset(
        data_root=config.data_root,
        split='val',
        transform=get_clip_transforms(),
        use_cached_embeddings=False  # We'll load them separately
    )
    
    image_paths = [val_dataset.image_dir / val_dataset.coco.imgs[img_id]['file_name'] 
                   for img_id in val_dataset.image_ids]
    captions = [cap for caps in val_dataset.captions for cap in caps]
    
    # Load cached embeddings if available
    cache_dir = Path(config.data_root) / config.cache_dir
    image_embeds_path = cache_dir / 'image_embeddings_val.pt'
    text_embeds_path = cache_dir / 'text_embeddings_val.pt'
    
    if image_embeds_path.exists():
        print(f"Loading cached image embeddings from {image_embeds_path}")
        image_embeds = torch.load(image_embeds_path)
    else:
        print("Warning: No cached image embeddings found. Will compute on-the-fly.")
        image_embeds = None
    
    if text_embeds_path.exists():
        print(f"Loading cached text embeddings from {text_embeds_path}")
        text_embeds = torch.load(text_embeds_path)
    else:
        print("Warning: No cached text embeddings found.")
        text_embeds = None
    
    return image_paths, captions, image_embeds, text_embeds


def generate_text_to_image_examples(model, image_paths, image_embeds, processor, device, save_dir):
    """Generate text-to-image retrieval examples."""
    print("\n" + "="*80)
    print("Generating Text-to-Image Retrieval Examples")
    print("="*80)
    
    # Define diverse queries as required by the lab
    queries = [
        "sport",  # Lab example
        "a cat sitting on a couch",
        "a red car on the street",
        "people playing in the park",
        "food on a table",
        "a dog running",
        "city skyline at night",
        "beach with ocean waves"
    ]
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate individual examples
    for query in queries[:5]:  # First 5 for individual examples
        safe_name = query.replace(" ", "_").replace(",", "")[:30]
        save_path = save_dir / f"text_to_image_{safe_name}.png"
        
        print(f"\nQuery: '{query}'")
        visualize_text_to_image_retrieval(
            query_text=query,
            model=model,
            image_paths=image_paths,
            image_embeds=image_embeds,
            clip_processor=processor,
            device=device,
            top_k=5,
            save_path=save_path
        )
    
    # Generate grid of multiple queries
    grid_path = save_dir / "text_to_image_grid.png"
    print("\nGenerating retrieval grid...")
    create_retrieval_grid(
        queries=queries[:4],
        model=model,
        image_paths=image_paths,
        image_embeds=image_embeds,
        clip_processor=processor,
        device=device,
        images_per_query=3,
        save_path=grid_path
    )


def generate_zero_shot_examples(model, image_paths, processor, device, save_dir):
    """Generate zero-shot classification examples."""
    print("\n" + "="*80)
    print("Generating Zero-Shot Classification Examples")
    print("="*80)
    
    transform = get_clip_transforms()
    os.makedirs(save_dir, exist_ok=True)
    
    # Example 1: Animal classification (as per lab example)
    class_sets = [
        {
            'name': 'animal_landscape_person',
            'labels': ['a person', 'an animal', 'a landscape'],
            'description': 'Basic classification'
        },
        {
            'name': 'animals',
            'labels': ['a cat', 'a dog', 'a bird', 'a horse', 'a cow'],
            'description': 'Animal types'
        },
        {
            'name': 'vehicles',
            'labels': ['a car', 'a truck', 'a motorcycle', 'a bicycle', 'a bus'],
            'description': 'Vehicle types'
        },
        {
            'name': 'activities',
            'labels': ['playing sports', 'eating food', 'working on computer', 'sleeping', 'reading'],
            'description': 'Human activities'
        }
    ]
    
    # Use random validation images
    random.seed(42)
    sample_indices = random.sample(range(len(image_paths)), min(10, len(image_paths)))
    
    for i, class_set in enumerate(class_sets):
        # Use a different image for each class set
        if i < len(sample_indices):
            img_path = image_paths[sample_indices[i]]
            save_path = save_dir / f"zero_shot_{class_set['name']}.png"
            
            print(f"\nClassifying {img_path.name} with {class_set['description']}...")
            predicted_class, confidence = zero_shot_classification(
                query_image_path=img_path,
                class_labels=class_set['labels'],
                model=model,
                clip_processor=processor,
                transform=transform,
                device=device,
                save_path=save_path
            )
            
            print(f"Predicted: {predicted_class} ({confidence*100:.1f}% confidence)")


def generate_image_to_text_examples(model, image_paths, captions, text_embeds, device, save_dir):
    """Generate image-to-text retrieval examples."""
    print("\n" + "="*80)
    print("Generating Image-to-Text Retrieval Examples")
    print("="*80)
    
    transform = get_clip_transforms()
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample random images
    random.seed(42)
    sample_indices = random.sample(range(len(image_paths)), min(5, len(image_paths)))
    
    for i, idx in enumerate(sample_indices):
        img_path = image_paths[idx]
        save_path = save_dir / f"image_to_text_example_{i+1}.png"
        
        print(f"\nRetrieving captions for {img_path.name}...")
        visualize_image_to_text_retrieval(
            query_image_path=img_path,
            model=model,
            captions=captions,
            text_embeds=text_embeds,
            transform=transform,
            device=device,
            top_k=5,
            save_path=save_path
        )


def main():
    """Main function to generate all visualizations."""
    # Configuration
    config = Config()
    device = torch.device(config.device)
    
    print("\n" + "="*80)
    print("CLIP Visualization Generator for Lab 4")
    print("="*80)
    print(f"Device: {device}")
    print(f"Results directory: {config.results_dir}")
    
    # Create visualization directory
    vis_dir = Path(config.results_dir) / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load model
    checkpoint_path = Path(config.checkpoint_dir) / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python train.py")
        return
    
    model = load_model_and_data(checkpoint_path, config, device)
    
    # Load validation data
    image_paths, captions, image_embeds, text_embeds = load_validation_data(config)
    print(f"\nLoaded {len(image_paths)} images and {len(captions)} captions")
    
    # Load CLIP processor for text encoding
    processor = CLIPProcessor.from_pretrained(config.clip_model_name)
    
    # Generate visualizations
    try:
        # 1. Text-to-Image Retrieval (Required by lab)
        text_to_image_dir = vis_dir / "text_to_image"
        generate_text_to_image_examples(
            model, image_paths, image_embeds, processor, device, text_to_image_dir
        )
        
        # 2. Zero-Shot Classification (Required by lab)
        zero_shot_dir = vis_dir / "zero_shot"
        generate_zero_shot_examples(
            model, image_paths, processor, device, zero_shot_dir
        )
        
        # 3. Image-to-Text Retrieval (Additional examples)
        if text_embeds is not None:
            image_to_text_dir = vis_dir / "image_to_text"
            generate_image_to_text_examples(
                model, image_paths, captions, text_embeds, device, image_to_text_dir
            )
        
        print("\n" + "="*80)
        print("Visualization Generation Complete!")
        print("="*80)
        print(f"\nAll visualizations saved to: {vis_dir}")
        print("\nGenerated files:")
        print(f"  - Text-to-Image retrieval: {text_to_image_dir}")
        print(f"  - Zero-shot classification: {zero_shot_dir}")
        if text_embeds is not None:
            print(f"  - Image-to-Text retrieval: {image_to_text_dir}")
        print("\nUse these images in your lab report!")
        
    except Exception as e:
        print(f"\nError during visualization generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualizations for CLIP lab report")
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: checkpoints/best_model.pth)')
    
    args = parser.parse_args()
    
    # Override checkpoint path if provided
    if args.checkpoint:
        # TODO: Pass to main function
        pass
    
    main()
