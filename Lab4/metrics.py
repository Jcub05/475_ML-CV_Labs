"""
Evaluation metrics for CLIP model.
Computes Recall@K for image-text retrieval tasks.
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
    Compute Recall@K for both image-to-text and text-to-image retrieval.
    
    Recall@K measures: "Is the correct match in the top-K retrieved items?"
    
    Args:
        similarity_matrix: Similarity matrix [N, N] where diagonal are ground truth pairs
        k_values: List of K values to compute recall for
        
    Returns:
        Dictionary with metrics:
            - img2txt_r1, img2txt_r5, img2txt_r10: Image-to-text recall
            - txt2img_r1, txt2img_r5, txt2img_r10: Text-to-image recall
    """
    n = similarity_matrix.shape[0]
    
    # Convert to numpy for easier indexing
    sim_matrix = similarity_matrix.cpu().numpy()
    
    metrics = {}
    
    # Image-to-Text Retrieval
    # For each image (row), rank all texts by similarity
    for k in k_values:
        # Get top-K text indices for each image
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]  # [N, K]
        
        # Check if correct text (diagonal index) is in top-K
        correct_indices = np.arange(n)[:, None]  # [N, 1]
        correct_in_top_k = np.any(top_k_indices == correct_indices, axis=1)  # [N]
        
        # Recall@K = percentage of queries where correct match is in top-K
        recall = correct_in_top_k.mean() * 100
        metrics[f'img2txt_r{k}'] = recall
    
    # Text-to-Image Retrieval
    # For each text (column), rank all images by similarity
    # Transpose the similarity matrix
    sim_matrix_t = sim_matrix.T
    
    for k in k_values:
        # Get top-K image indices for each text
        top_k_indices = np.argsort(-sim_matrix_t, axis=1)[:, :k]  # [N, K]
        
        # Check if correct image (diagonal index) is in top-K
        correct_indices = np.arange(n)[:, None]  # [N, 1]
        correct_in_top_k = np.any(top_k_indices == correct_indices, axis=1)  # [N]
        
        # Recall@K
        recall = correct_in_top_k.mean() * 100
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
    
    # Compute Recall@K metrics
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


def compute_mean_rank(similarity_matrix: torch.Tensor) -> Dict[str, float]:
    """
    Compute mean rank of correct matches.
    Lower is better (rank 1 = correct match is top result).
    
    Args:
        similarity_matrix: Similarity matrix [N, N]
        
    Returns:
        Dictionary with mean ranks
    """
    n = similarity_matrix.shape[0]
    sim_matrix = similarity_matrix.cpu().numpy()
    
    # Image-to-text ranks
    img2txt_ranks = []
    for i in range(n):
        # Get ranking of texts for image i
        ranking = np.argsort(-sim_matrix[i])  # Higher similarity = lower rank
        
        # Find where the correct text (index i) appears in ranking
        rank = np.where(ranking == i)[0][0] + 1  # +1 for 1-indexed rank
        img2txt_ranks.append(rank)
    
    # Text-to-image ranks
    txt2img_ranks = []
    for i in range(n):
        # Get ranking of images for text i
        ranking = np.argsort(-sim_matrix[:, i])
        
        # Find where the correct image (index i) appears
        rank = np.where(ranking == i)[0][0] + 1
        txt2img_ranks.append(rank)
    
    return {
        'img2txt_mean_rank': np.mean(img2txt_ranks),
        'txt2img_mean_rank': np.mean(txt2img_ranks),
        'img2txt_median_rank': np.median(img2txt_ranks),
        'txt2img_median_rank': np.median(txt2img_ranks)
    }


if __name__ == "__main__":
    # Test metrics
    print("\nTesting Retrieval Metrics\n")
    print("=" * 80)
    
    # Create dummy embeddings
    n_samples = 100
    embed_dim = 512
    
    print(f"Creating {n_samples} dummy image-text pairs...")
    
    # Random embeddings
    image_embeds = torch.randn(n_samples, embed_dim)
    text_embeds = torch.randn(n_samples, embed_dim)
    
    # Normalize
    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    
    # Test similarity computation
    print("\nComputing similarity matrix...")
    similarity = compute_similarity_matrix(image_embeds, text_embeds)
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")
    
    # Test Recall@K
    print("\nComputing Recall@K metrics...")
    metrics = compute_retrieval_metrics(image_embeds, text_embeds, k_values=[1, 5, 10])
    
    print("\nRandom embeddings (baseline):")
    for key, value in metrics.items():
        if 'recall' in key.lower():
            print(f"  {key}: {value:.2f}%")
    
    # Test with perfect matches
    print("\n" + "=" * 80)
    print("Testing with perfect matches...")
    
    perfect_embeds = torch.randn(n_samples, embed_dim)
    perfect_embeds = F.normalize(perfect_embeds, p=2, dim=-1)
    
    metrics_perfect = compute_retrieval_metrics(
        perfect_embeds, perfect_embeds, k_values=[1, 5, 10]
    )
    
    print("\nPerfect matches (should be 100%):")
    for key, value in metrics_perfect.items():
        if 'r' in key and 'avg' not in key:
            print(f"  {key}: {value:.2f}%")
    
    # Test retrieval function
    print("\n" + "=" * 80)
    print("Testing top-K retrieval...")
    
    query = perfect_embeds[0]
    database = perfect_embeds
    
    top_indices, top_sims = retrieve_top_k(query, database, k=5)
    print(f"\nTop-5 indices for query 0: {top_indices.tolist()}")
    print(f"Top-5 similarities: {top_sims.tolist()}")
    print(f"Expected: index 0 should be first (self-match)")
    
    # Test mean rank
    print("\n" + "=" * 80)
    print("Computing mean ranks...")
    
    rank_metrics = compute_mean_rank(similarity)
    print("\nRank metrics (random embeddings):")
    for key, value in rank_metrics.items():
        print(f"  {key}: {value:.2f}")
    
    rank_metrics_perfect = compute_mean_rank(
        compute_similarity_matrix(perfect_embeds, perfect_embeds)
    )
    print("\nRank metrics (perfect matches):")
    for key, value in rank_metrics_perfect.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n✓ Metrics test complete!")
    print("=" * 80)
