"""
Generate all required visualizations for CLIP Lab 4 (Section 2.4).

Requirements:
1. Text→Image retrieval: Given text query (e.g., "sport"), show top-5 retrieved images
2. Image classification: Given image + class list, classify the image

This script generates visualizations for BOTH:
- Fine-tuned model (your trained model)
- Base model (frozen OpenAI CLIP for comparison)

Usage:
    python generate_required_visualizations.py --model_path checkpoints/best_model.pth --val_dir path/to/val2014
    
    # For base model only (no fine-tuned model needed):
    python generate_required_visualizations.py --val_dir path/to/val2014 --base_only
"""

import torch
import argparse
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image

from model import CLIPFineTuneModel
from visualize import (
    visualize_text_to_image_retrieval,
    zero_shot_classification,
    create_retrieval_grid
)


def get_clip_transform():
    """Get CLIP preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])


def load_base_clip_model(device):
    """Load the base OpenAI CLIP model (frozen, no fine-tuning)."""
    print("Loading base OpenAI CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    print("✓ Base CLIP model loaded successfully")
    return model


def load_finetuned_model(model_path, device):
    """Load trained fine-tuned CLIP model."""
    print(f"Loading fine-tuned model from {model_path}...")
    
    model = CLIPFineTuneModel(
        embed_dim=512,
        pretrained_resnet=True,
        clip_model_name="openai/clip-vit-base-patch32",
        freeze_text_encoder=True
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("✓ Fine-tuned model loaded successfully")
    return model


def precompute_image_embeddings_finetuned(model, image_paths, transform, device, batch_size=32):
    """Precompute embeddings for all validation images using fine-tuned model."""
    print(f"\nPrecomputing embeddings for {len(image_paths)} images (fine-tuned model)...")
    
    all_embeds = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load images
            images = []
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                images.append(transform(img))
            
            images = torch.stack(images).to(device)
            
            # Encode using fine-tuned model
            embeds = model.encode_image(images).cpu()
            all_embeds.append(embeds)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i+len(batch_paths)}/{len(image_paths)} images")
    
    all_embeds = torch.cat(all_embeds, dim=0)
    print(f"✓ Precomputed embeddings: {all_embeds.shape}")
    return all_embeds


def precompute_image_embeddings_base(model, image_paths, processor, device, batch_size=32):
    """Precompute embeddings for all validation images using base CLIP model."""
    print(f"\nPrecomputing embeddings for {len(image_paths)} images (base CLIP)...")
    
    all_embeds = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load and preprocess images using CLIP processor
            images = []
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                images.append(img)
            
            # Use CLIP processor (it handles resizing and normalization)
            inputs = processor(images=images, return_tensors="pt").to(device)
            
            # Encode using base CLIP vision model
            embeds = model.get_image_features(**inputs)
            # Normalize embeddings
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i+len(batch_paths)}/{len(image_paths)} images")
    
    all_embeds = torch.cat(all_embeds, dim=0)
    print(f"✓ Precomputed embeddings: {all_embeds.shape}")
    return all_embeds


# Wrapper class to make base CLIP compatible with visualize.py functions
class BaseCLIPWrapper:
    """Wrapper to make base CLIP model compatible with visualization functions."""
    def __init__(self, clip_model, processor):
        self.clip_model = clip_model
        self.processor = processor
        
    def eval(self):
        self.clip_model.eval()
        return self
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text using base CLIP."""
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Normalize
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs
    
    def encode_image(self, images):
        """Encode images using base CLIP."""
        with torch.no_grad():
            # Base CLIP expects PIL images, but we get tensors
            # We need to use the vision model directly
            outputs = self.clip_model.vision_model(pixel_values=images).pooler_output
            # Project to embedding space
            outputs = self.clip_model.visual_projection(outputs)
            # Normalize
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs


def generate_visualizations_for_model(model, model_name, image_paths, image_embeds, 
                                     processor, transform, device, output_dir):
    """Generate all required visualizations for a given model."""
    
    print("\n" + "="*70)
    print(f"Generating visualizations for: {model_name}")
    print("="*70)
    
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # REQUIREMENT 1: Text→Image Retrieval
    # =========================================================================
    
    print("\nREQUIREMENT 1: Text→Image Retrieval")
    print("-" * 70)
    
    text_queries = [
        "sport",                 # Required example from lab
        "a dog playing",         # User's example
        "a person eating",
        "a beautiful sunset",
        "a cat on a couch"
    ]
    
    for query in text_queries:
        print(f"  Query: '{query}'")
        save_path = model_output_dir / f"text2img_{query.replace(' ', '_')}.png"
        
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
    
    # Create a grid view of multiple queries
    print(f"  Creating retrieval grid...")
    grid_path = model_output_dir / "text2img_grid.png"
    create_retrieval_grid(
        queries=text_queries[:4],  # Use first 4 for grid
        model=model,
        image_paths=image_paths,
        image_embeds=image_embeds,
        clip_processor=processor,
        device=device,
        images_per_query=5,
        save_path=grid_path
    )
    
    # =========================================================================
    # REQUIREMENT 2: Zero-Shot Image Classification
    # =========================================================================
    
    print("\nREQUIREMENT 2: Zero-Shot Image Classification")
    print("-" * 70)
    
    # Use the exact class list from the lab requirement
    class_labels = ['a person', 'an animal', 'a landscape']
    
    # Classify several example images
    num_classification_examples = 5
    sample_images = image_paths[:num_classification_examples]
    
    for idx, img_path in enumerate(sample_images):
        print(f"  Classifying image {idx+1}/{num_classification_examples}: {img_path.name}")
        save_path = model_output_dir / f"classification_example_{idx+1}.png"
        
        predicted_class, confidence = zero_shot_classification(
            query_image_path=img_path,
            class_labels=class_labels,
            model=model,
            clip_processor=processor,
            transform=transform,
            device=device,
            save_path=save_path
        )
        
        print(f"    → Predicted: {predicted_class} ({confidence*100:.1f}%)")
    
    print(f"\n✓ All visualizations for {model_name} saved to: {model_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for CLIP models")
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation images directory')
    parser.add_argument('--output_dir', type=str, default='Visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_images', type=int, default=1000,
                        help='Number of validation images to use')
    parser.add_argument('--base_only', action='store_true',
                        help='Only generate base model visualizations')
    parser.add_argument('--finetuned_only', action='store_true',
                        help='Only generate fine-tuned model visualizations')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("CLIP Lab 4 - Complete Visualization Generator")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Load CLIP processor and transform
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    transform = get_clip_transform()
    
    # Get validation images
    val_dir = Path(args.val_dir)
    all_image_paths = sorted(list(val_dir.glob("*.jpg")))[:args.num_images]
    print(f"\n✓ Found {len(all_image_paths)} validation images")
    
    # =========================================================================
    # Generate visualizations for BASE MODEL
    # =========================================================================
    
    if not args.finetuned_only:
        print("\n" + "="*70)
        print("PART 1: BASE CLIP MODEL (OpenAI Frozen)")
        print("="*70)
        
        base_clip = load_base_clip_model(device)
        base_clip_wrapped = BaseCLIPWrapper(base_clip, processor)
        
        # Precompute image embeddings for base model
        base_image_embeds = precompute_image_embeddings_base(
            base_clip, all_image_paths, processor, device
        )
        
        # Generate visualizations for base model
        generate_visualizations_for_model(
            model=base_clip_wrapped,
            model_name="base_model",
            image_paths=all_image_paths,
            image_embeds=base_image_embeds,
            processor=processor,
            transform=transform,
            device=device,
            output_dir=output_dir
        )
        
        # Clean up
        del base_clip, base_clip_wrapped, base_image_embeds
        torch.cuda.empty_cache()
    
    # =========================================================================
    # Generate visualizations for FINE-TUNED MODEL
    # =========================================================================
    
    if not args.base_only:
        if args.model_path is None:
            print("\n⚠ Warning: No --model_path provided. Skipping fine-tuned model.")
            print("   To generate fine-tuned model visualizations, provide --model_path")
        else:
            print("\n" + "="*70)
            print("PART 2: FINE-TUNED MODEL (Your Trained Model)")
            print("="*70)
            
            finetuned_model = load_finetuned_model(args.model_path, device)
            
            # Precompute image embeddings for fine-tuned model
            finetuned_image_embeds = precompute_image_embeddings_finetuned(
                finetuned_model, all_image_paths, transform, device
            )
            
            # Generate visualizations for fine-tuned model
            generate_visualizations_for_model(
                model=finetuned_model,
                model_name="finetuned_model",
                image_paths=all_image_paths,
                image_embeds=finetuned_image_embeds,
                processor=processor,
                transform=transform,
                device=device,
                output_dir=output_dir
            )
            
            # Clean up
            del finetuned_model, finetuned_image_embeds
            torch.cuda.empty_cache()
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION GENERATION COMPLETE")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    print(f"\nGenerated visualizations:")
    
    for subdir in output_dir.iterdir():
        if subdir.is_dir():
            num_files = len(list(subdir.glob("*.png")))
            print(f"\n  {subdir.name}/ ({num_files} images)")
            for file_path in sorted(subdir.glob("*.png"))[:5]:  # Show first 5
                size_kb = file_path.stat().st_size / 1024
                print(f"    • {file_path.name} ({size_kb:.1f} KB)")
            if num_files > 5:
                print(f"    ... and {num_files - 5} more")
    
    print("\n" + "="*70)
    print("These visualizations satisfy Lab 4 Section 2.4 requirements:")
    print("  ✓ Text→Image retrieval (including 'sport' and 'a dog playing')")
    print("  ✓ Image classification with ['a person', 'an animal', 'a landscape']")
    print("  ✓ Comparison between base and fine-tuned models")
    print("="*70)


if __name__ == "__main__":
    main()
