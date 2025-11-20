"""
Visualization utilities for CLIP model.
Includes text-to-image retrieval and zero-shot classification demos.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Optional
import numpy as np

from transformers import CLIPProcessor
from model import CLIPFineTuneModel
from metrics import retrieve_top_k


def visualize_text_to_image_retrieval(
    query_text: str,
    model: CLIPFineTuneModel,
    image_paths: List[Path],
    image_embeds: torch.Tensor,
    clip_processor: CLIPProcessor,
    device: str = "cuda",
    top_k: int = 5,
    save_path: Optional[Path] = None
):
    """
    Visualize top-K images retrieved for a text query.
    
    Args:
        query_text: Text query (e.g., "a cat")
        model: CLIP model
        image_paths: List of paths to all images
        image_embeds: Pre-computed image embeddings [N, embed_dim]
        clip_processor: CLIP processor for tokenizing text
        device: Device
        top_k: Number of images to retrieve
        save_path: Path to save visualization
    """
    model.eval()
    
    # Encode query text
    with torch.no_grad():
        inputs = clip_processor(
            text=[query_text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        text_embed = model.encode_text(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        text_embed = text_embed.cpu()
    
    # Retrieve top-K images
    top_indices, top_sims = retrieve_top_k(text_embed, image_embeds, k=top_k)
    
    # Visualize
    fig, axes = plt.subplots(1, top_k, figsize=(4 * top_k, 4))
    if top_k == 1:
        axes = [axes]
    
    for i, (idx, sim) in enumerate(zip(top_indices, top_sims)):
        # Load image
        img_path = image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Display
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Rank {i+1}\nSim: {sim:.3f}', fontsize=12)
    
    plt.suptitle(f'Query: "{query_text}"', fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_image_to_text_retrieval(
    query_image_path: Path,
    model: CLIPFineTuneModel,
    captions: List[str],
    text_embeds: torch.Tensor,
    transform,
    device: str = "cuda",
    top_k: int = 5,
    save_path: Optional[Path] = None
):
    """
    Visualize top-K captions retrieved for an image.
    
    Args:
        query_image_path: Path to query image
        model: CLIP model
        captions: List of all captions
        text_embeds: Pre-computed text embeddings [N, embed_dim]
        transform: Image transform
        device: Device
        top_k: Number of captions to retrieve
        save_path: Path to save visualization
    """
    model.eval()
    
    # Load and encode image
    img = Image.open(query_image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        img_embed = model.encode_image(img_tensor)
        img_embed = img_embed.cpu()
    
    # Retrieve top-K captions
    top_indices, top_sims = retrieve_top_k(img_embed, text_embeds, k=top_k)
    
    # Visualize
    fig = plt.figure(figsize=(12, 8))
    
    # Show image
    ax_img = plt.subplot(1, 2, 1)
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title('Query Image', fontsize=14)
    
    # Show top captions
    ax_text = plt.subplot(1, 2, 2)
    ax_text.axis('off')
    
    text_str = "Top Retrieved Captions:\n\n"
    for i, (idx, sim) in enumerate(zip(top_indices, top_sims)):
        caption = captions[idx]
        # Wrap long captions
        if len(caption) > 60:
            caption = caption[:57] + "..."
        text_str += f"{i+1}. [{sim:.3f}] {caption}\n\n"
    
    ax_text.text(0.1, 0.9, text_str, fontsize=11, verticalalignment='top',
                 family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def zero_shot_classification(
    query_image_path: Path,
    class_labels: List[str],
    model: CLIPFineTuneModel,
    clip_processor: CLIPProcessor,
    transform,
    device: str = "cuda",
    save_path: Optional[Path] = None
):
    """
    Perform zero-shot classification on an image.
    
    Args:
        query_image_path: Path to image
        class_labels: List of class names (e.g., ["a cat", "a dog", "a car"])
        model: CLIP model
        clip_processor: CLIP processor
        transform: Image transform
        device: Device
        save_path: Path to save visualization
    """
    model.eval()
    
    # Load and encode image
    img = Image.open(query_image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        img_embed = model.encode_image(img_tensor)
    
    # Encode class labels
    with torch.no_grad():
        inputs = clip_processor(
            text=class_labels,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        text_embeds = model.encode_text(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    
    # Compute similarities
    similarities = (img_embed @ text_embeds.T).squeeze(0)
    
    # Convert to probabilities
    probs = F.softmax(similarities * 100, dim=0).cpu().numpy()  # Scale for sharper probs
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    
    # Visualize
    fig = plt.figure(figsize=(12, 6))
    
    # Show image
    ax_img = plt.subplot(1, 2, 1)
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title('Query Image', fontsize=14)
    
    # Show predictions
    ax_bar = plt.subplot(1, 2, 2)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(class_labels))
    bars = ax_bar.barh(y_pos, probs[sorted_indices], color='steelblue', alpha=0.8)
    
    # Highlight top prediction
    bars[0].set_color('coral')
    
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([class_labels[i] for i in sorted_indices])
    ax_bar.set_xlabel('Probability', fontsize=12)
    ax_bar.set_title('Zero-Shot Classification', fontsize=14)
    ax_bar.set_xlim([0, 1])
    
    # Add percentage labels
    for i, (idx, prob) in enumerate(zip(sorted_indices, probs[sorted_indices])):
        ax_bar.text(prob + 0.02, i, f'{prob*100:.1f}%', 
                    va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print results
    print("\nZero-Shot Classification Results:")
    print("=" * 50)
    for i, idx in enumerate(sorted_indices):
        print(f"{i+1}. {class_labels[idx]:20s} {probs[idx]*100:5.2f}%")
    print("=" * 50)
    
    return class_labels[sorted_indices[0]], probs[sorted_indices[0]]


def create_retrieval_grid(
    queries: List[str],
    model: CLIPFineTuneModel,
    image_paths: List[Path],
    image_embeds: torch.Tensor,
    clip_processor: CLIPProcessor,
    device: str = "cuda",
    images_per_query: int = 3,
    save_path: Optional[Path] = None
):
    """
    Create a grid showing multiple text queries and their top retrieved images.
    
    Args:
        queries: List of text queries
        model: CLIP model
        image_paths: List of image paths
        image_embeds: Pre-computed image embeddings
        clip_processor: CLIP processor
        device: Device
        images_per_query: Number of images to show per query
        save_path: Path to save visualization
    """
    model.eval()
    
    num_queries = len(queries)
    
    fig = plt.figure(figsize=(images_per_query * 3, num_queries * 3))
    
    for q_idx, query in enumerate(queries):
        # Encode query
        with torch.no_grad():
            inputs = clip_processor(
                text=[query],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            text_embed = model.encode_text(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            ).cpu()
        
        # Retrieve top images
        top_indices, top_sims = retrieve_top_k(text_embed, image_embeds, k=images_per_query)
        
        # Display images
        for i, (idx, sim) in enumerate(zip(top_indices, top_sims)):
            ax = plt.subplot(num_queries, images_per_query, q_idx * images_per_query + i + 1)
            
            img = Image.open(image_paths[idx]).convert('RGB')
            ax.imshow(img)
            ax.axis('off')
            
            if i == 0:
                ax.set_title(f'"{query}"\n(sim: {sim:.2f})', fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'sim: {sim:.2f}', fontsize=9)
    
    plt.suptitle('Text-to-Image Retrieval Examples', fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("\nVisualization utilities for CLIP model")
    print("Import and use these functions with your trained model.")
    print("\nExample usage:")
    print("""
    from visualize import visualize_text_to_image_retrieval
    from model import CLIPFineTuneModel
    from transformers import CLIPProcessor
    
    # Load model and processor
    model = CLIPFineTuneModel(...)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Visualize retrieval
    visualize_text_to_image_retrieval(
        query_text="a cat playing",
        model=model,
        image_paths=my_image_paths,
        image_embeds=my_image_embeds,
        clip_processor=processor,
        top_k=5,
        save_path="retrieval_demo.png"
    )
    """)
