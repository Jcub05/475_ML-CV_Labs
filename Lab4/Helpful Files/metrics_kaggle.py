"""
Optimized evaluation metrics for CLIP model - Kaggle version.
Uses GPU-accelerated top-k instead of CPU numpy sorting.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


def compute_similarity_matrix(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity matrix between images and texts.
    
    Args:
        image_embeds: Image embeddings [N_img, embed_dim]
        text_embeds: Text embeddings [N_txt, embed_dim]
        
    Returns:
        Similarity matrix [N_img, N_txt]
    """
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    
    # Compute cosine similarity
    similarity = image_embeds @ text_embeds.T
    
    return similarity


def compute_recall_at_k(
    similarity_matrix: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute Recall@K using GPU-accelerated top-k (MUCH faster than numpy argsort).
    
    Args:
        similarity_matrix: Similarity matrix [N, N] where diagonal are ground truth pairs
        k_values: List of K values to compute recall for
        
    Returns:
        Dictionary with metrics
    """
    n = similarity_matrix.shape[0]
    device = similarity_matrix.device
    
    metrics = {}
    max_k = max(k_values)
    
    # Image-to-Text Retrieval
    # Get top-max_k indices for each image (on GPU - FAST!)
    _, top_k_indices = torch.topk(similarity_matrix, k=max_k, dim=1)  # [N, max_k]
    
    # Ground truth: diagonal indices
    correct_indices = torch.arange(n, device=device).unsqueeze(1)  # [N, 1]
    
    for k in k_values:
        # Check if correct index is in top-k
        top_k_for_this_k = top_k_indices[:, :k]  # [N, k]
        correct_in_top_k = (top_k_for_this_k == correct_indices).any(dim=1)  # [N]
        
        # Recall@K
        recall = correct_in_top_k.float().mean().item() * 100
        metrics[f'img2txt_r{k}'] = recall
    
    # Text-to-Image Retrieval
    # Transpose similarity matrix
    similarity_matrix_t = similarity_matrix.T
    
    # Get top-max_k indices for each text
    _, top_k_indices = torch.topk(similarity_matrix_t, k=max_k, dim=1)  # [N, max_k]
    
    for k in k_values:
        # Check if correct index is in top-k
        top_k_for_this_k = top_k_indices[:, :k]  # [N, k]
        correct_in_top_k = (top_k_for_this_k == correct_indices).any(dim=1)  # [N]
        
        # Recall@K
        recall = correct_in_top_k.float().mean().item() * 100
        metrics[f'txt2img_r{k}'] = recall
    
    return metrics


def compute_retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute full retrieval metrics given image and text embeddings.
    
    Args:
        image_embeds: Image embeddings [N, embed_dim]
        text_embeds: Text embeddings [N, embed_dim]
        k_values: List of K values for Recall@K
        
    Returns:
        Dictionary with all retrieval metrics
    """
    # Compute similarity matrix
    similarity = compute_similarity_matrix(image_embeds, text_embeds)
    
    # Compute Recall@K metrics (GPU-accelerated)
    metrics = compute_recall_at_k(similarity, k_values=k_values)
    
    # Add average metrics
    avg_img2txt = np.mean([metrics[f'img2txt_r{k}'] for k in k_values])
    avg_txt2img = np.mean([metrics[f'txt2img_r{k}'] for k in k_values])
    avg_both = (avg_img2txt + avg_txt2img) / 2
    
    metrics['avg_img2txt_recall'] = avg_img2txt
    metrics['avg_txt2img_recall'] = avg_txt2img
    metrics['avg_recall'] = avg_both
    
    return metrics


def retrieve_top_k(
    query_embed: torch.Tensor,
    database_embeds: torch.Tensor,
    k: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve top-K most similar items from database.
    
    Args:
        query_embed: Query embedding [embed_dim] or [1, embed_dim]
        database_embeds: Database embeddings [N, embed_dim]
        k: Number of items to retrieve
        
    Returns:
        Tuple of (top_k_indices, top_k_similarities)
    """
    # Ensure query is 2D
    if query_embed.dim() == 1:
        query_embed = query_embed.unsqueeze(0)
    
    # Normalize
    query_embed = F.normalize(query_embed, p=2, dim=-1)
    database_embeds = F.normalize(database_embeds, p=2, dim=-1)
    
    # Compute similarities
    similarities = query_embed @ database_embeds.T  # [1, N]
    similarities = similarities.squeeze(0)  # [N]
    
    # Get top-K
    top_k_values, top_k_indices = torch.topk(similarities, k=min(k, len(similarities)))
    
    return top_k_indices, top_k_values
