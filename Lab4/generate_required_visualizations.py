"""
Generate all required visualizations for CLIP Lab 4 (Section 2.4).

Requirements:
1. Text→Image retrieval: Given text query (e.g., "sport"), show top-5 retrieved images
2. Image classification: Given image + class list, classify the image

Usage:
    python generate_required_visualizations.py --model_path best_model.pth --val_dir path/to/val2014
"""

import torch
import argparse
from pathlib import Path
from transformers import CLIPProcessor
from torchvision import transforms

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


def load_model(model_path, device):
    """Load trained CLIP model."""
    print(f"Loading model from {model_path}...")
    
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
    
    print("✓ Model loaded successfully")
    return model


def precompute_image_embeddings(model, image_paths, transform, device, batch_size=32):
    """Precompute embeddings for all validation images."""
    print(f"\nPrecomputing embeddings for {len(image_paths)} images...")
    
    all_embeds = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load images
            from PIL import Image
            images = []
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                images.append(transform(img))
            
            images = torch.stack(images).to(device)
            
            # Encode
            embeds = model.encode_image(images).cpu()
            all_embeds.append(embeds)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i+len(batch_paths)}/{len(image_paths)} images")
    
    all_embeds = torch.cat(all_embeds, dim=0)
    print(f"✓ Precomputed embeddings: {all_embeds.shape}")
    return all_embeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation images directory')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_images', type=int, default=1000,
                        help='Number of validation images to use')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("CLIP Lab 4 - Required Visualizations Generator")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Load model
    model = load_model(args.model_path, device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    transform = get_clip_transform()
    
    # Get validation images
    val_dir = Path(args.val_dir)
    all_image_paths = sorted(list(val_dir.glob("*.jpg")))[:args.num_images]
    print(f"\n✓ Found {len(all_image_paths)} validation images")
    
    # Precompute image embeddings
    image_embeds = precompute_image_embeddings(model, all_image_paths, transform, device)
    
    # =========================================================================
    # REQUIREMENT 1: Text→Image Retrieval
    # Lab requirement: "Given a text query (such as 'sport'), display the top-5 retrieved images"
    # =========================================================================
    
    print("\n" + "="*70)
    print("REQUIREMENT 1: Text→Image Retrieval")
    print("="*70)
    
    text_queries = [
        "sport",           # Required example from lab
        "a dog",
        "a person eating",
        "a beautiful sunset"
    ]
    
    for query in text_queries:
        print(f"\n  Query: '{query}'")
        save_path = output_dir / f"text2img_{query.replace(' ', '_')}.png"
        
        visualize_text_to_image_retrieval(
            query_text=query,
            model=model,
            image_paths=all_image_paths,
            image_embeds=image_embeds,
            clip_processor=processor,
            device=device,
            top_k=5,
            save_path=save_path
        )
    
    # Also create a grid view of multiple queries
    print(f"\n  Creating retrieval grid...")
    grid_path = output_dir / "text2img_grid.png"
    create_retrieval_grid(
        queries=text_queries,
        model=model,
        image_paths=all_image_paths,
        image_embeds=image_embeds,
        clip_processor=processor,
        device=device,
        images_per_query=5,
        save_path=grid_path
    )
    
    # =========================================================================
    # REQUIREMENT 2: Image Classification
    # Lab requirement: "Given an image and a list of classes (such as 
    # ['a person', 'an animal', 'a landscape']), classify the image"
    # =========================================================================
    
    print("\n" + "="*70)
    print("REQUIREMENT 2: Zero-Shot Image Classification")
    print("="*70)
    
    # Use the exact class list from the lab requirement
    class_labels = ['a person', 'an animal', 'a landscape']
    
    # Classify several example images
    num_classification_examples = 5
    sample_images = all_image_paths[:num_classification_examples]
    
    for idx, img_path in enumerate(sample_images):
        print(f"\n  Classifying image {idx+1}/{num_classification_examples}: {img_path.name}")
        save_path = output_dir / f"classification_example_{idx+1}.png"
        
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
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "="*70)
    print("✅ ALL REQUIRED VISUALIZATIONS GENERATED")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    
    all_outputs = sorted(output_dir.glob("*.png"))
    for file_path in all_outputs:
        size_kb = file_path.stat().st_size / 1024
        print(f"  • {file_path.name} ({size_kb:.1f} KB)")
    
    print("\n" + "="*70)
    print("These visualizations satisfy Section 2.4 requirements:")
    print("  ✓ Text→Image retrieval with 'sport' and other queries")
    print("  ✓ Image classification with ['a person', 'an animal', 'a landscape']")
    print("="*70)


if __name__ == "__main__":
    main()
