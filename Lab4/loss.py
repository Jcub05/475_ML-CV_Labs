"""
InfoNCE Loss for CLIP Training.
Contrastive loss that aligns image and text embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) Loss for CLIP.
    
    This loss maximizes similarity between matching image-text pairs (positives)
    while minimizing similarity with non-matching pairs (negatives).
    
    Mathematical formulation:
    For a batch of N image-text pairs, we compute:
    1. Similarity matrix S = image_embeds @ text_embeds.T  [N x N]
    2. Apply temperature scaling: S = S / temperature
    3. Image-to-text loss: -log(exp(S[i,i]) / sum_j(exp(S[i,j])))
    4. Text-to-image loss: -log(exp(S[i,i]) / sum_j(exp(S[j,i])))
    5. Final loss = (i2t_loss + t2i_loss) / 2
    
    The diagonal elements S[i,i] are the positive pairs.
    All off-diagonal elements are negatives.
    """
    
    def __init__(self, temperature: float = 0.07, learnable_temperature: bool = False):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for scaling (default: 0.07)
            learnable_temperature: Whether to make temperature a learnable parameter
        """
        super().__init__()
        
        if learnable_temperature:
            # Initialize as log(1/temperature) for numerical stability
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature)))
        else:
            self.register_buffer('logit_scale', torch.tensor(1.0 / temperature))
        
        self.learnable_temperature = learnable_temperature
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            image_embeds: Normalized image embeddings [B, embed_dim]
            text_embeds: Normalized text embeddings [B, embed_dim]
            
        Returns:
            Scalar loss value
        """
        # Compute cosine similarity matrix
        # Both embeddings are L2-normalized, so this is just dot product
        logits = image_embeds @ text_embeds.T  # [B, B]
        
        # Apply temperature scaling
        if self.learnable_temperature:
            logit_scale = torch.exp(self.logit_scale)
            logits = logits * logit_scale
        else:
            logits = logits * self.logit_scale
        
        # Labels: diagonal elements are the positive pairs
        # For a batch of size B, label[i] = i
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # Image-to-text loss (rows are images, columns are texts)
        # For each image, the correct text is on the diagonal
        loss_i2t = F.cross_entropy(logits, labels)
        
        # Text-to-image loss (columns are images, rows are texts)
        # We transpose the logits matrix
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # Symmetric loss (average of both directions)
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        if self.learnable_temperature:
            return 1.0 / torch.exp(self.logit_scale).item()
        else:
            return 1.0 / self.logit_scale.item()


class InfoNCELossWithMetrics(InfoNCELoss):
    """
    InfoNCE loss that also computes accuracy metrics.
    Useful for monitoring training progress.
    """
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        return_metrics: bool = False
    ):
        """
        Compute loss and optionally return metrics.
        
        Args:
            image_embeds: Image embeddings [B, embed_dim]
            text_embeds: Text embeddings [B, embed_dim]
            return_metrics: Whether to compute and return accuracy metrics
            
        Returns:
            If return_metrics=False: loss (scalar)
            If return_metrics=True: (loss, metrics_dict)
        """
        # Compute similarity matrix
        logits = image_embeds @ text_embeds.T
        
        # Apply temperature
        if self.learnable_temperature:
            logit_scale = torch.exp(self.logit_scale)
            logits = logits * logit_scale
        else:
            logits = logits * self.logit_scale
        
        # Labels
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # Losses
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2.0
        
        if not return_metrics:
            return loss
        
        # Compute accuracy metrics
        with torch.no_grad():
            # Image-to-text accuracy (top-1)
            i2t_pred = logits.argmax(dim=1)
            i2t_acc = (i2t_pred == labels).float().mean().item()
            
            # Text-to-image accuracy (top-1)
            t2i_pred = logits.T.argmax(dim=1)
            t2i_acc = (t2i_pred == labels).float().mean().item()
            
            # Average similarity of positive pairs
            pos_sim = logits.diagonal().mean().item()
            
            # Average similarity of negative pairs
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=logits.device)
            neg_sim = logits[mask].mean().item()
        
        metrics = {
            'loss': loss.item(),
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item(),
            'i2t_acc': i2t_acc * 100,  # Convert to percentage
            't2i_acc': t2i_acc * 100,
            'pos_sim': pos_sim,
            'neg_sim': neg_sim,
            'temperature': self.get_temperature()
        }
        
        return loss, metrics


if __name__ == "__main__":
    # Test InfoNCE loss
    print("\nTesting InfoNCE Loss\n")
    print("=" * 80)
    
    # Create dummy embeddings
    batch_size = 8
    embed_dim = 512
    
    # Random embeddings (normalized)
    image_embeds = torch.randn(batch_size, embed_dim)
    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    
    text_embeds = torch.randn(batch_size, embed_dim)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    
    # Test basic loss
    print("Testing basic InfoNCE loss...")
    criterion = InfoNCELoss(temperature=0.07)
    loss = criterion(image_embeds, text_embeds)
    print(f"Loss: {loss.item():.4f}")
    print(f"Temperature: {criterion.get_temperature():.4f}")
    
    # Test with learnable temperature
    print("\nTesting with learnable temperature...")
    criterion_learnable = InfoNCELoss(temperature=0.07, learnable_temperature=True)
    loss_learnable = criterion_learnable(image_embeds, text_embeds)
    print(f"Loss: {loss_learnable.item():.4f}")
    print(f"Initial temperature: {criterion_learnable.get_temperature():.4f}")
    
    # Test loss with metrics
    print("\nTesting InfoNCE loss with metrics...")
    criterion_metrics = InfoNCELossWithMetrics(temperature=0.07)
    loss, metrics = criterion_metrics(image_embeds, text_embeds, return_metrics=True)
    
    print(f"\nMetrics:")
    for key, value in metrics.items():
        if 'acc' in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:.4f}")
    
    # Test with perfect matches
    print("\n" + "=" * 80)
    print("Testing with perfect positive matches...")
    perfect_embeds = torch.randn(batch_size, embed_dim)
    perfect_embeds = F.normalize(perfect_embeds, p=2, dim=-1)
    
    loss_perfect = criterion(perfect_embeds, perfect_embeds)
    loss_perfect_metrics, metrics_perfect = criterion_metrics(
        perfect_embeds, perfect_embeds, return_metrics=True
    )
    
    print(f"Loss with perfect matches: {loss_perfect.item():.4f}")
    print(f"i2t accuracy: {metrics_perfect['i2t_acc']:.2f}%")
    print(f"t2i accuracy: {metrics_perfect['t2i_acc']:.2f}%")
    
    print("\n✓ Loss test complete!")
    print("=" * 80)
