"""
Figure 2 Analysis: Cosine Similarity between Direction Vectors

This script implements the analysis for Figure 2 from the paper:
"Frame-level Temporal Difference Learning for Partial Deepfake Speech Detection"

Implements Equation (1) and Equation (2):
    - Equation (1): Δx_t = (x_{t+1} - x_t) / ||x_{t+1} - x_t||
    - Equation (2): cos(θ_t) = Δx_t · Δx_{t+1}

Computes mean (μ) and standard deviation (σ) of cosine similarities for each utterance.
"""

import os
import torch
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from temporal_difference_learning import temporal_difference_learning


def compute_cosine_similarity_stats_from_embeddings(embedding_dir: str) -> tuple:
    """
    Compute cosine similarity statistics from saved embeddings.

    This implements Equations (1) and (2) for Figure 2 of the paper.

    Args:
        embedding_dir: Directory containing saved embeddings
                      (e.g., /nvme3/wj/embeddings/)

    Returns:
        Tuple of two dictionaries:
        - mean_results: {"bonafide": [μ1, μ2, ...], "A01": [...], ...}
        - std_results:  {"bonafide": [σ1, σ2, ...], "A01": [...], ...}

    Formula:
        For each utterance with embeddings X of shape (T, D):
            Δx_t = (x_{t+1} - x_t) / ||x_{t+1} - x_t||  (Equation 1)
            cos(θ_t) = Δx_t · Δx_{t+1}                  (Equation 2)
            μ = mean(cos(θ_t))
            σ = std(cos(θ_t))
    """
    print("="*80)
    print("Figure 2: Cosine Similarity Analysis (Equation 1 & 2)")
    print("="*80)

    mean_results = {}
    std_results = {}

    # Process each attack type
    attack_types = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]

    for attack_type in attack_types:
        attack_dir = os.path.join(embedding_dir, attack_type)

        if not os.path.exists(attack_dir):
            print(f"[WARNING] Directory not found: {attack_dir}")
            continue

        # Get all .pt files
        pt_files = list(Path(attack_dir).glob("*.pt"))

        if len(pt_files) == 0:
            print(f"[WARNING] No embeddings found for {attack_type}")
            continue

        print(f"\n[INFO] Processing {attack_type}: {len(pt_files)} files")

        means = []
        stds = []

        for pt_file in tqdm(pt_files, desc=f"{attack_type}", ncols=100):
            try:
                # Load embedding (T, 1024)
                embeddings = torch.load(pt_file, map_location='cpu')

                # Check minimum frames needed
                if embeddings.shape[0] < 3:
                    # Need at least 3 frames to compute cosine similarity
                    # (T=3 -> 2 direction vectors -> 1 cosine similarity)
                    continue

                # Apply Equation 1 & 2
                direction_vectors, cosine_similarities = temporal_difference_learning(embeddings)

                # Compute μ and σ
                μ = cosine_similarities.mean().item()
                σ = cosine_similarities.std().item()

                means.append(μ)
                stds.append(σ)

            except Exception as e:
                print(f"\n[ERROR] Failed to process {pt_file.name}: {e}")
                continue

        mean_results[attack_type] = means
        std_results[attack_type] = stds

        print(f"[INFO] {attack_type}: Processed {len(means)} utterances")
        print(f"       Mean of μ: {np.mean(means):.4f}, Std of μ: {np.std(means):.4f}")
        print(f"       Mean of σ: {np.mean(stds):.4f}, Std of σ: {np.std(stds):.4f}")

    return mean_results, std_results


def plot_figure2_boxplots(mean_results: dict, std_results: dict, output_dir: str, figsize=(14, 8)):
    """
    Create two horizontal boxplots for Figure 2 (mean and std deviation).

    Args:
        mean_results: Dictionary of mean values from compute_cosine_similarity_stats_from_embeddings()
        std_results: Dictionary of std values from compute_cosine_similarity_stats_from_embeddings()
        output_dir: Directory to save plots
        figsize: Figure size (width, height)
    """
    print("\n" + "="*80)
    print("Creating Figure 2 Boxplots (Horizontal)")
    print("="*80)

    # REVERSED ORDER (bonafide at top)
    attack_types = ["A06", "A05", "A04", "A03", "A02", "A01", "bonafide"]

    # Attack type labels with TTS/VC annotation
    attack_labels = {
        "bonafide": "Bonafide",
        "A01": "A01-TTS",
        "A02": "A02-TTS",
        "A03": "A03-TTS",
        "A04": "A04-TTS",
        "A05": "A05-VC",
        "A06": "A06-VC"
    }

    # Prepare data for mean
    mean_data = []
    mean_labels = []
    for attack_type in attack_types:
        if attack_type in mean_results and len(mean_results[attack_type]) > 0:
            mean_data.append(mean_results[attack_type])
            mean_labels.append(attack_labels[attack_type])

    # Prepare data for std
    std_data = []
    std_labels = []
    for attack_type in attack_types:
        if attack_type in std_results and len(std_results[attack_type]) > 0:
            std_data.append(std_results[attack_type])
            std_labels.append(attack_labels[attack_type])

    # Debug: print data shapes
    print(f"[DEBUG] mean_data length: {len(mean_data)}, mean_labels length: {len(mean_labels)}")
    print(f"[DEBUG] std_data length: {len(std_data)}, std_labels length: {len(std_labels)}")
    print(f"[DEBUG] mean_labels: {mean_labels}")
    print(f"[DEBUG] std_labels: {std_labels}")
    for i, (label, data) in enumerate(zip(mean_labels, mean_data)):
        print(f"[DEBUG] mean_data[{i}] ({label}): {len(data) if hasattr(data, '__len__') else 'scalar'} values")

    # Create figure with two subplots (vertical stacking)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Colors
    colors_mean = ['#e74c3c' if 'TTS' in label or 'VC' in label else '#2ecc71' for label in mean_labels]
    colors_std = ['#e74c3c' if 'TTS' in label or 'VC' in label else '#2ecc71' for label in std_labels]

    # Plot 1: Mean (μ) - HORIZONTAL
    bp1 = ax1.boxplot(mean_data, labels=mean_labels, patch_artist=True, showfliers=True, vert=False)
    for patch, color in zip(bp1['boxes'], colors_mean):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_xlabel("Mean Cosine Similarity (μ)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Attack Type", fontsize=11, fontweight='bold')
    ax1.set_title("(a) Mean of cos(θ)", fontsize=12, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='both', labelsize=10)

    # Plot 2: Standard Deviation (σ) - HORIZONTAL
    bp2 = ax2.boxplot(std_data, labels=std_labels, patch_artist=True, showfliers=True, vert=False)
    for patch, color in zip(bp2['boxes'], colors_std):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_xlabel("Std Deviation of Cosine Similarity (σ)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Attack Type", fontsize=11, fontweight='bold')
    ax2.set_title("(b) Std Deviation of cos(θ)", fontsize=12, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax2.set_axisbelow(True)
    ax2.tick_params(axis='both', labelsize=10)

    # Main title
    fig.suptitle("Figure 2: Cosine Similarity Statistics (Temporal Difference Learning)",
                 fontsize=14, fontweight='bold', y=0.995)

    # Tight layout
    plt.tight_layout()

    # Save combined figure
    combined_path = os.path.join(output_dir, "figure2_cosine_similarity_boxplots.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Combined figure saved to: {combined_path}")

    # Save as PDF
    pdf_path = combined_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Combined figure saved to: {pdf_path}")

    plt.close()

    # Also save individual plots
    # Plot mean only - HORIZONTAL (with reversed order and TTS/VC labels)
    # Prepare individual plot data (same reversed order)
    individual_attack_types = ["A06", "A05", "A04", "A03", "A02", "A01", "bonafide"]
    individual_attack_labels = {
        "bonafide": "Bonafide",
        "A01": "A01-TTS",
        "A02": "A02-TTS",
        "A03": "A03-TTS",
        "A04": "A04-TTS",
        "A05": "A05-VC",
        "A06": "A06-VC"
    }

    individual_mean_data = []
    individual_mean_labels = []
    for attack_type in individual_attack_types:
        if attack_type in mean_results and len(mean_results[attack_type]) > 0:
            individual_mean_data.append(mean_results[attack_type])
            individual_mean_labels.append(individual_attack_labels[attack_type])

    individual_std_data = []
    individual_std_labels = []
    for attack_type in individual_attack_types:
        if attack_type in std_results and len(std_results[attack_type]) > 0:
            individual_std_data.append(std_results[attack_type])
            individual_std_labels.append(individual_attack_labels[attack_type])

    individual_colors_mean = ['#e74c3c' if 'TTS' in label or 'VC' in label else '#2ecc71' for label in individual_mean_labels]
    individual_colors_std = ['#e74c3c' if 'TTS' in label or 'VC' in label else '#2ecc71' for label in individual_std_labels]

    fig_mean, ax_mean = plt.subplots(figsize=(10, 8))
    bp_mean = ax_mean.boxplot(individual_mean_data, labels=individual_mean_labels, patch_artist=True, showfliers=True, vert=False)
    for patch, color in zip(bp_mean['boxes'], individual_colors_mean):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax_mean.set_xlabel("Mean Cosine Similarity (μ)", fontsize=12, fontweight='bold')
    ax_mean.set_ylabel("Attack Type", fontsize=12, fontweight='bold')
    ax_mean.set_title("Figure 2(a): Mean of Cosine Similarity", fontsize=14, fontweight='bold', pad=20)
    ax_mean.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax_mean.set_axisbelow(True)
    plt.tight_layout()

    mean_path = os.path.join(output_dir, "figure2_mean_boxplot.png")
    plt.savefig(mean_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Mean boxplot saved to: {mean_path}")
    plt.close()

    # Plot std only - HORIZONTAL
    fig_std, ax_std = plt.subplots(figsize=(10, 8))
    bp_std = ax_std.boxplot(individual_std_data, labels=individual_std_labels, patch_artist=True, showfliers=True, vert=False)
    for patch, color in zip(bp_std['boxes'], individual_colors_std):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax_std.set_xlabel("Std Deviation of Cosine Similarity (σ)", fontsize=12, fontweight='bold')
    ax_std.set_ylabel("Attack Type", fontsize=12, fontweight='bold')
    ax_std.set_title("Figure 2(b): Std Deviation of Cosine Similarity", fontsize=14, fontweight='bold', pad=20)
    ax_std.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax_std.set_axisbelow(True)
    plt.tight_layout()

    std_path = os.path.join(output_dir, "figure2_std_boxplot.png")
    plt.savefig(std_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Std boxplot saved to: {std_path}")
    plt.close()

    # =========================================================================
    # Additional Plot: Bonafide vs Deepfake (All attacks combined)
    # =========================================================================
    print("\n[INFO] Creating Bonafide vs Deepfake comparison plot...")

    # Combine all deepfake attacks (A01-A06)
    deepfake_mean_data = []
    deepfake_std_data = []

    for attack_type in ["A01", "A02", "A03", "A04", "A05", "A06"]:
        if attack_type in mean_results and len(mean_results[attack_type]) > 0:
            deepfake_mean_data.extend(mean_results[attack_type])
        if attack_type in std_results and len(std_results[attack_type]) > 0:
            deepfake_std_data.extend(std_results[attack_type])

    # Get bonafide data
    bonafide_mean_data = mean_results.get("bonafide", [])
    bonafide_std_data = std_results.get("bonafide", [])

    # Create comparison figure (2 subplots: mean and std)
    fig_compare, (ax_mean_cmp, ax_std_cmp) = plt.subplots(2, 1, figsize=(10, 10))

    # Bonafide vs Deepfake - Mean
    if len(bonafide_mean_data) > 0 and len(deepfake_mean_data) > 0:
        bp_mean_cmp = ax_mean_cmp.boxplot(
            [deepfake_mean_data, bonafide_mean_data],
            labels=["Deepfake", "Bonafide"],
            patch_artist=True,
            showfliers=True,
            vert=False
        )

        # Colors: red for deepfake, green for bonafide
        colors_cmp = ['#e74c3c', '#2ecc71']
        for patch, color in zip(bp_mean_cmp['boxes'], colors_cmp):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax_mean_cmp.set_xlabel("Mean Cosine Similarity (μ)", fontsize=12, fontweight='bold')
        ax_mean_cmp.set_ylabel("Type", fontsize=12, fontweight='bold')
        ax_mean_cmp.set_title("(a) Bonafide vs Deepfake: Mean of cos(θ)", fontsize=13, fontweight='bold', pad=15)
        ax_mean_cmp.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax_mean_cmp.set_axisbelow(True)

    # Bonafide vs Deepfake - Std
    if len(bonafide_std_data) > 0 and len(deepfake_std_data) > 0:
        bp_std_cmp = ax_std_cmp.boxplot(
            [deepfake_std_data, bonafide_std_data],
            labels=["Deepfake", "Bonafide"],
            patch_artist=True,
            showfliers=True,
            vert=False
        )

        # Colors: red for deepfake, green for bonafide
        colors_cmp = ['#e74c3c', '#2ecc71']
        for patch, color in zip(bp_std_cmp['boxes'], colors_cmp):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax_std_cmp.set_xlabel("Std Deviation of Cosine Similarity (σ)", fontsize=12, fontweight='bold')
        ax_std_cmp.set_ylabel("Type", fontsize=12, fontweight='bold')
        ax_std_cmp.set_title("(b) Bonafide vs Deepfake: Std Deviation of cos(θ)", fontsize=13, fontweight='bold', pad=15)
        ax_std_cmp.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax_std_cmp.set_axisbelow(True)

    # Main title
    fig_compare.suptitle("Figure 2: Bonafide vs Deepfake Comparison",
                         fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save comparison figure
    compare_path = os.path.join(output_dir, "figure2_bonafide_vs_deepfake.png")
    plt.savefig(compare_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Bonafide vs Deepfake comparison saved to: {compare_path}")

    compare_pdf_path = compare_path.replace('.png', '.pdf')
    plt.savefig(compare_pdf_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Bonafide vs Deepfake comparison saved to: {compare_pdf_path}")

    plt.close()


def print_statistics(mean_results: dict, std_results: dict):
    """
    Print detailed statistics for each attack type.

    Args:
        mean_results: Dictionary of mean values
        std_results: Dictionary of std values
    """
    print("\n" + "="*80)
    print("Statistical Summary")
    print("="*80)

    attack_types = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]

    print("\n" + "-"*80)
    print("Mean Cosine Similarity (μ)")
    print("-"*80)
    print(f"{'Attack Type':<12} {'Count':>8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 80)

    for attack_type in attack_types:
        if attack_type in mean_results and len(mean_results[attack_type]) > 0:
            values = mean_results[attack_type]
            print(f"{attack_type:<12} {len(values):>8} "
                  f"{np.mean(values):>12.6f} {np.std(values):>12.6f} "
                  f"{np.min(values):>12.6f} {np.max(values):>12.6f}")
        else:
            print(f"{attack_type:<12} {'N/A':>8}")

    print("\n" + "-"*80)
    print("Standard Deviation of Cosine Similarity (σ)")
    print("-"*80)
    print(f"{'Attack Type':<12} {'Count':>8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 80)

    for attack_type in attack_types:
        if attack_type in std_results and len(std_results[attack_type]) > 0:
            values = std_results[attack_type]
            print(f"{attack_type:<12} {len(values):>8} "
                  f"{np.mean(values):>12.6f} {np.std(values):>12.6f} "
                  f"{np.min(values):>12.6f} {np.max(values):>12.6f}")
        else:
            print(f"{attack_type:<12} {'N/A':>8}")


def save_results(mean_results: dict, std_results: dict, output_path: str):
    """
    Save results to pickle file.

    Args:
        mean_results: Dictionary of mean values
        std_results: Dictionary of std values
        output_path: Path to save pickle file
    """
    results = {
        'mean': mean_results,
        'std': std_results
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"[INFO] Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Figure 2 Analysis: Cosine Similarity (Equation 1 & 2)"
    )
    parser.add_argument(
        '--embedding_dir',
        type=str,
        default='/nvme3/wj/embeddings/',
        help='Directory containing saved embeddings'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/',
        help='Directory to save results and plots'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("Figure 2 Analysis: Cosine Similarity (Temporal Difference Learning)")
    print("="*80)
    print(f"Embedding directory: {args.embedding_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Compute cosine similarity statistics
    mean_results, std_results = compute_cosine_similarity_stats_from_embeddings(args.embedding_dir)

    # Print statistics
    print_statistics(mean_results, std_results)

    # Save results
    result_path = os.path.join(args.output_dir, "figure2_results.pkl")
    save_results(mean_results, std_results, result_path)

    # Plot boxplots
    plot_figure2_boxplots(mean_results, std_results, args.output_dir)

    print("\n" + "="*80)
    print("✓ Figure 2 Analysis Complete!")
    print("="*80)
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
