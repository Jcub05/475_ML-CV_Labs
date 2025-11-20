"""
Modified CLIP Image Encoder Architectures for Ablation Study
Each modification can be independently enabled/disabled for systematic comparison.
"""

import torch
import torch.nn as nn
from torchvision import models


class CLIPImageEncoderModified(nn.Module):
    """
    Modified CLIP Image Encoder with optional architectural improvements:
    1. BatchNorm in projection head
    2. Dropout regularization
    3. Deeper projection head (3 layers instead of 2)
    4. Learnable temperature parameter
    """
    
    def __init__(self, 
                 embed_dim=512,
                 use_batchnorm=False,
                 use_dropout=False,
                 dropout_rate=0.1,
                 deeper_projection=False,
                 learnable_temperature=False,
                 initial_temperature=0.07):
        """
        Args:
            embed_dim: Dimension of the embedding space
            use_batchnorm: Add BatchNorm layers in projection head
            use_dropout: Add Dropout layers in projection head
            dropout_rate: Dropout probability
            deeper_projection: Use 3-layer projection instead of 2-layer
            learnable_temperature: Make temperature a learnable parameter
            initial_temperature: Initial value for temperature
        """
        super(CLIPImageEncoderModified, self).__init__()
        
        # Store configuration
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.deeper_projection = deeper_projection
        self.learnable_temperature = learnable_temperature
        
        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Get the output dimension of ResNet50 (2048)
        resnet_output_dim = 2048
        
        # Build projection head based on configuration
        if deeper_projection:
            # 3-layer projection: 2048 -> 1024 -> 512 -> embed_dim
            layers = [nn.Linear(resnet_output_dim, 1024)]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(1024))
            layers.append(nn.GELU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            
            layers.append(nn.Linear(1024, 512))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(512))
            layers.append(nn.GELU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            
            layers.append(nn.Linear(512, embed_dim))
        else:
            # Standard 2-layer projection: 2048 -> 2048 -> embed_dim
            layers = [nn.Linear(resnet_output_dim, resnet_output_dim)]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(resnet_output_dim))
            layers.append(nn.GELU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            
            layers.append(nn.Linear(resnet_output_dim, embed_dim))
        
        self.projection = nn.Sequential(*layers)
        
        # Temperature parameter
        if learnable_temperature:
            # Use log-space to ensure temperature stays positive
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(initial_temperature)))
        else:
            self.register_buffer('temperature', torch.tensor(initial_temperature))
    
    def forward(self, images):
        """
        Args:
            images: Tensor of shape (batch_size, 3, 224, 224)
        Returns:
            embeddings: L2-normalized embeddings of shape (batch_size, embed_dim)
        """
        # Extract features from ResNet50
        features = self.backbone(images)  # (batch_size, 2048, 1, 1)
        features = features.flatten(1)     # (batch_size, 2048)
        
        # Project to embedding space
        embeddings = self.projection(features)  # (batch_size, embed_dim)
        
        # L2 normalization
        embeddings = nn.functional.normalize(embeddings, dim=-1)
        
        return embeddings
    
    def get_temperature(self):
        """Get the current temperature value"""
        if self.learnable_temperature:
            return torch.exp(self.log_temperature)
        else:
            return self.temperature


class CLIPFineTuneModelModified(nn.Module):
    """
    Complete CLIP model for fine-tuning with modified image encoder.
    Text encoder remains frozen.
    """
    
    def __init__(self, 
                 image_encoder,
                 text_encoder,
                 tokenizer):
        """
        Args:
            image_encoder: Modified CLIPImageEncoderModified instance
            text_encoder: Frozen CLIP text encoder from HuggingFace
            tokenizer: CLIP tokenizer from HuggingFace
        """
        super(CLIPFineTuneModelModified, self).__init__()
        
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def encode_text(self, captions):
        """
        Encode text captions to embeddings.
        
        Args:
            captions: List of strings or pre-tokenized dict
        Returns:
            text_embeddings: L2-normalized embeddings of shape (batch_size, embed_dim)
        """
        # Tokenize if needed
        if isinstance(captions, list):
            inputs = self.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(next(self.text_encoder.parameters()).device)
        else:
            inputs = captions
        
        # Get text features
        outputs = self.text_encoder(**inputs)
        text_embeddings = outputs.pooler_output  # (batch_size, 512)
        
        # L2 normalization
        text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)
        
        return text_embeddings
    
    def encode_image(self, images):
        """
        Encode images to embeddings.
        
        Args:
            images: Tensor of shape (batch_size, 3, 224, 224)
        Returns:
            image_embeddings: L2-normalized embeddings of shape (batch_size, embed_dim)
        """
        return self.image_encoder(images)
    
    def forward(self, images, text_embeddings=None, captions=None):
        """
        Forward pass for computing image and text embeddings.
        
        Args:
            images: Tensor of shape (batch_size, 3, 224, 224)
            text_embeddings: Pre-computed text embeddings (batch_size, embed_dim)
            captions: List of caption strings (if text_embeddings not provided)
        Returns:
            image_embeddings, text_embeddings, temperature
        """
        # Encode images
        image_embeddings = self.encode_image(images)
        
        # Get text embeddings
        if text_embeddings is None:
            if captions is None:
                raise ValueError("Either text_embeddings or captions must be provided")
            text_embeddings = self.encode_text(captions)
        
        # Get temperature
        temperature = self.image_encoder.get_temperature()
        
        return image_embeddings, text_embeddings, temperature


def create_modified_model(text_encoder, 
                         tokenizer,
                         embed_dim=512,
                         use_batchnorm=False,
                         use_dropout=False,
                         dropout_rate=0.1,
                         deeper_projection=False,
                         learnable_temperature=False,
                         initial_temperature=0.07):
    """
    Factory function to create a modified CLIP model with specified modifications.
    
    Args:
        text_encoder: Frozen CLIP text encoder from HuggingFace
        tokenizer: CLIP tokenizer from HuggingFace
        embed_dim: Dimension of the embedding space
        use_batchnorm: Add BatchNorm layers in projection head
        use_dropout: Add Dropout layers in projection head
        dropout_rate: Dropout probability
        deeper_projection: Use 3-layer projection instead of 2-layer
        learnable_temperature: Make temperature a learnable parameter
        initial_temperature: Initial value for temperature
    
    Returns:
        CLIPFineTuneModelModified instance
    """
    image_encoder = CLIPImageEncoderModified(
        embed_dim=embed_dim,
        use_batchnorm=use_batchnorm,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        deeper_projection=deeper_projection,
        learnable_temperature=learnable_temperature,
        initial_temperature=initial_temperature
    )
    
    model = CLIPFineTuneModelModified(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer
    )
    
    return model


# Predefined model configurations for ablation study
MODEL_CONFIGS = {
    'baseline': {
        'use_batchnorm': False,
        'use_dropout': False,
        'deeper_projection': False,
        'learnable_temperature': False,
    },
    'batchnorm': {
        'use_batchnorm': True,
        'use_dropout': False,
        'deeper_projection': False,
        'learnable_temperature': False,
    },
    'dropout': {
        'use_batchnorm': False,
        'use_dropout': True,
        'dropout_rate': 0.1,
        'deeper_projection': False,
        'learnable_temperature': False,
    },
    'deeper': {
        'use_batchnorm': False,
        'use_dropout': False,
        'deeper_projection': True,
        'learnable_temperature': False,
    },
    'learnable_temp': {
        'use_batchnorm': False,
        'use_dropout': False,
        'deeper_projection': False,
        'learnable_temperature': True,
    },
    'batchnorm_dropout': {
        'use_batchnorm': True,
        'use_dropout': True,
        'dropout_rate': 0.1,
        'deeper_projection': False,
        'learnable_temperature': False,
    },
    'all_combined': {
        'use_batchnorm': True,
        'use_dropout': True,
        'dropout_rate': 0.1,
        'deeper_projection': True,
        'learnable_temperature': True,
    }
}
