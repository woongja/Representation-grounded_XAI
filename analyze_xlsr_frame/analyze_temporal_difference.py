"""
Temporal Difference Learning Analysis

This script:
1. Loads saved embeddings (T, 1024)
2. Applies Equation (1): Δx_t = (x_{t+1} - x_t) / ||x_{t+1} - x_t||
3. Applies Equation (2): cos(θ_t) = Δx_t · Δx_{t+1}
4. Computes μ (mean) and σ (std) per utterance
5. Plots results: bonafide vs deepfake
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle


# =============================================================================
# Equation (1): Direction Vectors
# =============================================================================
def compute_direction_vectors(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Equation (1): Δx_t = (x_{t+1} - x_t) / ||x_{t+1} - x_t||

    Args:
        X: Frame embeddings of shape (T, D)

    Returns:
        direction_vectors: Normalized direction vectors of shape (T-1, D)
    """
    if X.shape[0] < 2:
        raise ValueError(f"Need at least 2 frames, got {X.shape[0]}")

    # Compute frame differences: x_{t+1} - x_t
    frame_diffs = X[1:] - X[:-1]  # (T-1, D)

    # Compute norms
    norms = torch.norm(frame_diffs, p=2, dim=1, keepdim=True)  # (T-1, 1)

    # Normalize: Δx_t = (x_{t+1} - x_t) / ||x_{t+1} - x_t||
    direction_vectors = frame_diffs / (norms + eps)  # (T-1, D)

    return direction_vectors


# =============================================================================
# Equation (2): Cosine Similarity
# =============================================================================
def compute_cosine_similarities(direction_vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Equation (2): cos(θ_t) = Δx_t · Δx_{t+1}

    Args:
        direction_vectors: Normalized direction vectors of shape (T-1, D)

    Returns:
        cosine_similarities: Cosine similarities of shape (T-2,)
    """
    if direction_vectors.shape[0] < 2:
        raise ValueError(f"Need at least 2 direction vectors, got {direction_vectors.shape[0]}")

    # Get consecutive direction vectors
    delta_t = direction_vectors[:-1]        # Δx_t, shape: (T-2, D)
    delta_t_plus_1 = direction_vectors[1:]  # Δx_{t+1}, shape: (T-2, D)

    # Compute dot products: Δx_t · Δx_{t+1}
    dot_products = (delta_t * delta_t_plus_1).sum(dim=1)  # (T-2,)

    # For normalized vectors, cosine similarity = dot product
    # But we add robustness with explicit normalization
    norms_t = torch.norm(delta_t, p=2, dim=1)
    norms_t_plus_1 = torch.norm(delta_t_plus_1, p=2, dim=1)

    cosine_similarities = dot_products / (norms_t * norms_t_plus_1 + eps)

    return cosine_similarities


# =============================================================================
# Complete Pipeline
# =============================================================================
def temporal_difference_learning(X: torch.Tensor, eps: float = 1e-8):
    """
    Complete pipeline: Apply Equation (1) then Equation (2)

    Args:
        X: Frame embeddings of shape (T, D)

    Returns:
        direction_vectors: (T-1, D)
        cosine_similarities: (T-2,)
    """
    # Equation (1)
    direction_vectors = compute_direction_vectors(X, eps=eps)

    # Equation (2)
    cosine_similarities = compute_cosine_similarities(direction_vectors, eps=eps)

    return direction_vectors, cosine_similarities


# =============================================================================
# Compute Statistics per Utterance
# =============================================================================
def compute_statistics_from_embeddings(embedding_dir: str):
    """
    For each utterance:
        1. Load embeddings (T, 1024)
        2. Apply Equation (1) → direction vectors
        3. Apply Equation (2) → cosine similarities
        4. Compute μ = mean(cos(θ)) and σ = std(cos(θ))

    Returns:
        mean_results: {attack_type: [μ1, μ2, ...]}
        std_results: {attack_type: [σ1, σ2, ...]}
    """
    attack_types = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]
    mean_results = {}
    std_results = {}

    print("="*80)
    print("Temporal Difference Learning Analysis")
    print("="*80)

    for attack_type in attack_types:
        attack_dir = os.path.join(embedding_dir, attack_type)

        if not os.path.exists(attack_dir):
            print(f"[WARNING] Missing folder: {attack_type}")
            continue

        pt_files = list(Path(attack_dir).glob("*.pt"))
        if len(pt_files) == 0:
            print(f"[WARNING] No embeddings for {attack_type}")
            continue

        means = []
        stds = []

        print(f"\n[INFO] Processing {attack_type}: {len(pt_files)} files")

        for pt_file in tqdm(pt_files, desc=f"{attack_type:>10}"):
            try:
                # Load embeddings
                emb = torch.load(pt_file, map_location='cpu')  # (T, 1024)

                if emb.shape[0] < 3:
                    continue  # Need at least 3 frames

                # Apply Equation (1) + Equation (2)
                _, cos_vals = temporal_difference_learning(emb)  # (T-2,)

                # Compute statistics
                μ = cos_vals.mean().item()
                σ = cos_vals.std().item()

                means.append(μ)
                stds.append(σ)

            except Exception as e:
                print(f"[ERROR] {pt_file.name}: {e}")

        mean_results[attack_type] = means
        std_results[attack_type] = stds

    return mean_results, std_results


# =============================================================================
# Plot Results
# =============================================================================
def plot_results(mean_results, std_results, output_dir):
    """
    Create boxplots comparing bonafide vs deepfake
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    bonafide_mean = mean_results.get("bonafide", [])
    bonafide_std = std_results.get("bonafide", [])

    deepfake_mean = []
    deepfake_std = []
    for attack in ["A01", "A02", "A03", "A04", "A05", "A06"]:
        if attack in mean_results:
            deepfake_mean.extend(mean_results[attack])
            deepfake_std.extend(std_results[attack])

    if len(bonafide_mean) == 0 or len(deepfake_mean) == 0:
        print("[ERROR] Not enough data to plot")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2))

    mean_data = [bonafide_mean, deepfake_mean]
    std_data  = [bonafide_std, deepfake_std]

    # Plot mean (μ)
    bp1 = ax1.boxplot(
        mean_data,
        vert=False,
        labels=["Bonafide", "Deepfake"],
        patch_artist=True
    )
    bp1['boxes'][0].set_facecolor('#2ecc71')
    bp1['boxes'][1].set_facecolor('#e74c3c')
    for box in bp1['boxes']:
        box.set_alpha(0.6)

    ax1.set_title("Mean (μ) of cos(θ)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Mean Cosine Similarity (μ)")
    ax1.grid(alpha=0.3, linestyle="--")

    # Force Bonafide TOP, Deepfake BOTTOM
    ax1.invert_yaxis()


    # Plot std (σ)
    bp2 = ax2.boxplot(
        std_data,
        vert=False,
        labels=["Bonafide", "Deepfake"],
        patch_artist=True
    )
    bp2['boxes'][0].set_facecolor('#2ecc71')
    bp2['boxes'][1].set_facecolor('#e74c3c')
    for box in bp2['boxes']:
        box.set_alpha(0.6)

    ax2.set_title("Std (σ) of cos(θ)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Std Deviation of Cosine Similarity (σ)")
    ax2.grid(alpha=0.3, linestyle="--")

    # Force Bonafide TOP, Deepfake BOTTOM
    ax2.invert_yaxis()


    plt.tight_layout()

    # Save
    save_path = os.path.join(output_dir, "bonafide_vs_deepfake.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[INFO] Plot saved to: {save_path}")


# =============================================================================
# Print Statistics
# =============================================================================
def print_statistics(mean_results, std_results):
    """
    Print statistics table
    """
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"{'Attack':<10} {'μ_mean':>10} {'μ_std':>10} {'σ_mean':>10} {'σ_std':>10} {'Count':>8}")
    print("-"*80)

    attack_types = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]

    for attack in attack_types:
        if attack not in mean_results:
            continue

        μ_vals = np.array(mean_results[attack])
        σ_vals = np.array(std_results[attack])

        print(f"{attack:<10} "
              f"{μ_vals.mean():>10.4f} {μ_vals.std():>10.4f} "
              f"{σ_vals.mean():>10.4f} {σ_vals.std():>10.4f} "
              f"{len(μ_vals):>8}")

    # Deepfake combined
    deepfake_means = []
    deepfake_stds = []
    for attack in ["A01", "A02", "A03", "A04", "A05", "A06"]:
        if attack in mean_results:
            deepfake_means.extend(mean_results[attack])
            deepfake_stds.extend(std_results[attack])

    if len(deepfake_means) > 0:
        μ_vals = np.array(deepfake_means)
        σ_vals = np.array(deepfake_stds)

        print("-"*80)
        print(f"{'Deepfake':<10} "
              f"{μ_vals.mean():>10.4f} {μ_vals.std():>10.4f} "
              f"{σ_vals.mean():>10.4f} {σ_vals.std():>10.4f} "
              f"{len(μ_vals):>8}")

    print("="*80)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Temporal Difference Learning Analysis"
    )
    parser.add_argument(
        '--embedding_dir',
        type=str,
        required=True,
        help='Directory containing saved embeddings'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/',
        help='Directory to save results'
    )

    args = parser.parse_args()

    print("="*80)
    print("Temporal Difference Learning Analysis")
    print("="*80)
    print(f"Embedding dir: {args.embedding_dir}")
    print(f"Output dir: {args.output_dir}")
    print("="*80)

    # Compute statistics
    mean_results, std_results = compute_statistics_from_embeddings(args.embedding_dir)

    # Print statistics
    print_statistics(mean_results, std_results)

    # Plot results
    plot_results(mean_results, std_results, args.output_dir)

    # Save results
    result_path = os.path.join(args.output_dir, "results.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump({"mean": mean_results, "std": std_results}, f)
    print(f"[INFO] Results saved to: {result_path}")

    print("\n" + "="*80)
    print("✓ Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
