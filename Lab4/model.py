"""
CLIP Model Architecture for Fine-tuning.
- ResNet50 image encoder (pretrained on ImageNet)
- Frozen CLIP text encoder from HuggingFace
- Projection head to align embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import CLIPModel
from typing import Tuple


class CLIPImageEncoder(nn.Module):
    """
    Image encoder based on ResNet50 with projection head.
    Maps images to the CLIP embedding space (512-dim).
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize image encoder.
        
        Args:
            embed_dim: Dimension of output embeddings (CLIP uses 512)
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze ResNet backbone
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Load ResNet50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        # ResNet50 has 2048-dim features before the FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head: 2048 -> 512
        # Two-layer MLP with GELU activation
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, embed_dim)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Batch of images [B, 3, 224, 224]
            
        Returns:
            Normalized embeddings [B, embed_dim]
        """
        # Extract features from ResNet
        features = self.backbone(images)  # [B, 2048, 1, 1]
        features = features.flatten(1)     # [B, 2048]
        
        # Project to CLIP embedding space
        embeddings = self.projection_head(features)  # [B, 512]
        
        # L2 normalize (important for cosine similarity)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class CLIPFineTuneModel(nn.Module):
    """
    Complete CLIP model for fine-tuning.
    - Fine-tuned ResNet50 image encoder
    - Frozen CLIP text encoder
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        pretrained_resnet: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_text_encoder: bool = True
    ):
        """
        Initialize CLIP model.
        
        Args:
            embed_dim: Embedding dimension
            pretrained_resnet: Use ImageNet pretrained ResNet
            clip_model_name: HuggingFace CLIP model name
            freeze_text_encoder: Whether to freeze text encoder
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Image encoder (trainable)
        self.image_encoder = CLIPImageEncoder(
            embed_dim=embed_dim,
            pretrained=pretrained_resnet,
            freeze_backbone=False
        )
        
        # Text encoder (frozen)
        print(f"Loading CLIP text encoder from {clip_model_name}...")
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.text_encoder = clip_model.text_model
        self.text_projection = clip_model.text_projection
        
        # Freeze text encoder
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.text_projection.parameters():
                param.requires_grad = False
            self.text_encoder.eval()
        
        print("✓ Model initialized")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        return self.image_encoder(images)
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings using frozen CLIP encoder.
        
        Args:
            input_ids: Tokenized text [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            Normalized text embeddings [B, embed_dim]
        """
        # Get text features from CLIP
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool and project
        pooled_output = text_outputs.pooler_output
        text_embeds = self.text_projection(pooled_output)
        
        # Normalize
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        return text_embeds
    
    def forward(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Batch of images [B, 3, 224, 224]
            text_embeddings: Pre-computed text embeddings [B, 512] (if cached)
            input_ids: Tokenized text [B, seq_len] (if not cached)
            attention_mask: Attention mask [B, seq_len] (if not cached)
            
        Returns:
            Tuple of (image_embeddings, text_embeddings)
        """
        # Encode images
        image_embeds = self.encode_image(images)
        
        # Get text embeddings
        if text_embeddings is not None:
            # Use pre-computed embeddings (faster)
            text_embeds = text_embeddings
        elif input_ids is not None and attention_mask is not None:
            # Encode text on-the-fly
            text_embeds = self.encode_text(input_ids, attention_mask)
        else:
            raise ValueError("Either text_embeddings or (input_ids, attention_mask) must be provided")
        
        return image_embeds, text_embeds
    
    def compute_similarity(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity matrix between images and text.
        
        Args:
            image_embeds: Image embeddings [B, embed_dim]
            text_embeds: Text embeddings [B, embed_dim]
            
        Returns:
            Similarity matrix [B, B]
        """
        # Cosine similarity (embeddings are already normalized)
        similarity = image_embeds @ text_embeds.T
        return similarity


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    # Test model
    print("\nTesting CLIP Model Architecture\n")
    print("=" * 80)
    
    # Create model
    print("Creating model...")
    model = CLIPFineTuneModel(
        embed_dim=512,
        pretrained_resnet=True,
        clip_model_name="openai/clip-vit-base-patch32",
        freeze_text_encoder=True
    )
    
    # Count parameters
    trainable, total = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    print(f"  Trainable %: {100 * trainable / total:.2f}%")
    
    # Test forward pass with dummy data
    print("\n" + "=" * 80)
    print("Testing forward pass...")
    
    batch_size = 4
    
    # Dummy images
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Dummy pre-computed text embeddings (simulating cached)
    text_embeddings = torch.randn(batch_size, 512)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        image_embeds, text_embeds = model(images, text_embeddings=text_embeddings)
    
    print(f"\nOutput shapes:")
    print(f"  Image embeddings: {image_embeds.shape}")
    print(f"  Text embeddings: {text_embeds.shape}")
    
    # Test similarity computation
    similarity = model.compute_similarity(image_embeds, text_embeds)
    print(f"  Similarity matrix: {similarity.shape}")
    
    # Check normalization
    image_norms = torch.norm(image_embeds, p=2, dim=-1)
    print(f"\nEmbedding norms (should be ~1.0):")
    print(f"  Image: {image_norms.mean():.4f} ± {image_norms.std():.4f}")
    
    print("\n✓ Model test complete!")
    print("=" * 80)


def create_clip_model(text_encoder, tokenizer, embed_dim=512):
    """
    Factory function to create baseline CLIP model.
    Used by ablation_study.py for compatibility.
    
    Args:
        text_encoder: Frozen CLIP text encoder (not used, model loads its own)
        tokenizer: CLIP tokenizer (not used)
        embed_dim: Embedding dimension (default 512)
    
    Returns:
        CLIPFineTuneModel instance
    """
    model = CLIPFineTuneModel(
        embed_dim=embed_dim,
        pretrained_resnet=True,
        clip_model_name='openai/clip-vit-base-patch32',
        freeze_text_encoder=True
    )
    return model
