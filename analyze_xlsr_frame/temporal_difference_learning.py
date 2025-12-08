"""
Frame-level Temporal Difference Learning for Partial Deepfake Speech Detection

This module implements the mathematical operations from:
- Equation (1): Directional frame difference
- Equation (2): Cosine similarity between consecutive directions

Paper: "Frame-level Temporal Difference Learning for Partial Deepfake Speech Detection"
"""

import torch
import torch.nn.functional as F


def compute_direction_vectors(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute directional frame difference vectors (Equation 1).

    Equation (1):
        Δx_t = (x_{t+1} - x_t) / ||x_{t+1} - x_t||

    Args:
        X: Frame-level embeddings of shape (T, D)
           T = number of frames
           D = embedding dimension
        eps: Small epsilon value to prevent division by zero

    Returns:
        direction_vectors: Normalized direction vectors of shape (T-1, D)

    Example:
        >>> X = torch.randn(100, 1024)  # 100 frames, 1024-dim embeddings
        >>> delta_x = compute_direction_vectors(X)
        >>> delta_x.shape
        torch.Size([99, 1024])
    """
    if X.dim() != 2:
        raise ValueError(f"Expected 2D tensor (T, D), got shape {X.shape}")

    T, D = X.shape

    if T < 2:
        raise ValueError(f"Need at least 2 frames to compute directions, got {T}")

    # Compute frame differences: x_{t+1} - x_t
    # Shape: (T-1, D)
    frame_diffs = X[1:] - X[:-1]

    # Compute L2 norms for each difference vector
    # Shape: (T-1,)
    norms = torch.norm(frame_diffs, p=2, dim=1, keepdim=True)

    # Normalize: Δx_t = (x_{t+1} - x_t) / ||x_{t+1} - x_t||
    # Add eps to prevent division by zero
    direction_vectors = frame_diffs / (norms + eps)

    return direction_vectors


def compute_cosine_similarities(direction_vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute cosine similarities between consecutive direction vectors (Equation 2).

    Equation (2):
        cos(θ_t) = (Δx_t · Δx_{t+1}) / (||Δx_t|| * ||Δx_{t+1}||)

    Since direction_vectors are already normalized (||Δx_t|| = 1), this simplifies to:
        cos(θ_t) = Δx_t · Δx_{t+1}

    Args:
        direction_vectors: Normalized direction vectors of shape (T-1, D)
        eps: Small epsilon value for numerical stability (not strictly needed for normalized vectors)

    Returns:
        cosine_similarities: Cosine similarities of shape (T-2,)

    Example:
        >>> delta_x = torch.randn(99, 1024)
        >>> delta_x = F.normalize(delta_x, p=2, dim=1)  # Normalize
        >>> cos_theta = compute_cosine_similarities(delta_x)
        >>> cos_theta.shape
        torch.Size([98])
    """
    if direction_vectors.dim() != 2:
        raise ValueError(f"Expected 2D tensor (T-1, D), got shape {direction_vectors.shape}")

    T_minus_1, D = direction_vectors.shape

    if T_minus_1 < 2:
        raise ValueError(f"Need at least 2 direction vectors to compute cosine similarities, got {T_minus_1}")

    # Get consecutive direction vectors
    delta_t = direction_vectors[:-1]      # Δx_t, shape: (T-2, D)
    delta_t_plus_1 = direction_vectors[1:]  # Δx_{t+1}, shape: (T-2, D)

    # Compute dot products: Δx_t · Δx_{t+1}
    # Shape: (T-2,)
    dot_products = (delta_t * delta_t_plus_1).sum(dim=1)

    # If direction vectors are already normalized (as they should be from compute_direction_vectors),
    # the cosine similarity is just the dot product.
    # However, we can add extra normalization for robustness:
    norms_t = torch.norm(delta_t, p=2, dim=1)
    norms_t_plus_1 = torch.norm(delta_t_plus_1, p=2, dim=1)

    # cos(θ_t) = dot_product / (||Δx_t|| * ||Δx_{t+1}||)
    cosine_similarities = dot_products / (norms_t * norms_t_plus_1 + eps)

    return cosine_similarities


def temporal_difference_learning(X: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Complete temporal difference learning pipeline combining Equations (1) and (2).

    This function computes:
    1. Direction vectors (Equation 1): Δx_t = (x_{t+1} - x_t) / ||x_{t+1} - x_t||
    2. Cosine similarities (Equation 2): cos(θ_t) = (Δx_t · Δx_{t+1}) / (||Δx_t|| * ||Δx_{t+1}||)

    Args:
        X: Frame-level embeddings of shape (T, D)
           T = number of frames
           D = embedding dimension
        eps: Small epsilon value to prevent division by zero

    Returns:
        direction_vectors: Normalized direction vectors of shape (T-1, D)
        cosine_similarities: Cosine similarities of shape (T-2,)

    Example:
        >>> X = torch.randn(100, 1024)  # 100 frames, 1024-dim embeddings
        >>> delta_x, cos_theta = temporal_difference_learning(X)
        >>> delta_x.shape, cos_theta.shape
        (torch.Size([99, 1024]), torch.Size([98]))
    """
    # Equation (1): Compute direction vectors
    direction_vectors = compute_direction_vectors(X, eps=eps)

    # Equation (2): Compute cosine similarities
    cosine_similarities = compute_cosine_similarities(direction_vectors, eps=eps)

    return direction_vectors, cosine_similarities


# Batch processing version for efficiency
def temporal_difference_learning_batch(X_batch: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch version of temporal difference learning.

    Args:
        X_batch: Batch of frame-level embeddings of shape (B, T, D)
                 B = batch size
                 T = number of frames
                 D = embedding dimension
        eps: Small epsilon value to prevent division by zero

    Returns:
        direction_vectors_batch: Normalized direction vectors of shape (B, T-1, D)
        cosine_similarities_batch: Cosine similarities of shape (B, T-2)

    Example:
        >>> X_batch = torch.randn(8, 100, 1024)  # 8 samples, 100 frames, 1024-dim
        >>> delta_x_batch, cos_theta_batch = temporal_difference_learning_batch(X_batch)
        >>> delta_x_batch.shape, cos_theta_batch.shape
        (torch.Size([8, 99, 1024]), torch.Size([8, 98]))
    """
    if X_batch.dim() != 3:
        raise ValueError(f"Expected 3D tensor (B, T, D), got shape {X_batch.shape}")

    B, T, D = X_batch.shape

    if T < 2:
        raise ValueError(f"Need at least 2 frames to compute directions, got {T}")

    # Equation (1): Compute direction vectors for entire batch
    # Shape: (B, T-1, D)
    frame_diffs = X_batch[:, 1:, :] - X_batch[:, :-1, :]
    norms = torch.norm(frame_diffs, p=2, dim=2, keepdim=True)
    direction_vectors_batch = frame_diffs / (norms + eps)

    # Equation (2): Compute cosine similarities for entire batch
    if T >= 3:
        delta_t = direction_vectors_batch[:, :-1, :]      # (B, T-2, D)
        delta_t_plus_1 = direction_vectors_batch[:, 1:, :]  # (B, T-2, D)

        dot_products = (delta_t * delta_t_plus_1).sum(dim=2)  # (B, T-2)

        norms_t = torch.norm(delta_t, p=2, dim=2)
        norms_t_plus_1 = torch.norm(delta_t_plus_1, p=2, dim=2)

        cosine_similarities_batch = dot_products / (norms_t * norms_t_plus_1 + eps)
    else:
        # If T == 2, we only have 1 direction vector, can't compute cosine similarity
        cosine_similarities_batch = torch.empty(B, 0, device=X_batch.device)

    return direction_vectors_batch, cosine_similarities_batch


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("Temporal Difference Learning - Example Usage")
    print("="*80)

    # Single sample example
    print("\n1. Single Sample Example:")
    print("-" * 80)
    T, D = 100, 1024  # 100 frames, 1024-dimensional embeddings
    X = torch.randn(T, D)

    print(f"Input shape: {X.shape}")

    direction_vectors, cosine_similarities = temporal_difference_learning(X)

    print(f"Direction vectors shape: {direction_vectors.shape}")
    print(f"Cosine similarities shape: {cosine_similarities.shape}")
    print(f"\nCosine similarity statistics:")
    print(f"  Mean: {cosine_similarities.mean().item():.4f}")
    print(f"  Std:  {cosine_similarities.std().item():.4f}")
    print(f"  Min:  {cosine_similarities.min().item():.4f}")
    print(f"  Max:  {cosine_similarities.max().item():.4f}")

    # Batch example
    print("\n2. Batch Processing Example:")
    print("-" * 80)
    B, T, D = 8, 100, 1024  # 8 samples, 100 frames, 1024-dimensional
    X_batch = torch.randn(B, T, D)

    print(f"Input batch shape: {X_batch.shape}")

    direction_vectors_batch, cosine_similarities_batch = temporal_difference_learning_batch(X_batch)

    print(f"Direction vectors batch shape: {direction_vectors_batch.shape}")
    print(f"Cosine similarities batch shape: {cosine_similarities_batch.shape}")
    print(f"\nBatch cosine similarity statistics:")
    print(f"  Mean: {cosine_similarities_batch.mean().item():.4f}")
    print(f"  Std:  {cosine_similarities_batch.std().item():.4f}")
    print(f"  Min:  {cosine_similarities_batch.min().item():.4f}")
    print(f"  Max:  {cosine_similarities_batch.max().item():.4f}")

    # Verification: Direction vectors should be normalized
    print("\n3. Verification:")
    print("-" * 80)
    norms = torch.norm(direction_vectors, p=2, dim=1)
    print(f"Direction vector norms (should be ~1.0):")
    print(f"  Mean: {norms.mean().item():.6f}")
    print(f"  Std:  {norms.std().item():.6f}")

    print("\n" + "="*80)
