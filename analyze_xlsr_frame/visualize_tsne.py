"""
t-SNE Visualization of SSL Embeddings

This script:
1. Loads saved embeddings (T, 1024) from each attack type
2. Samples frames from embeddings
3. Applies t-SNE dimensionality reduction
4. Plots t-SNE visualization with different colors for each attack type
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle


# =============================================================================
# Load Embeddings
# =============================================================================
def load_embeddings_from_dir(embedding_dir: str, max_files_per_attack: int = None,
                             max_frames_per_file: int = 50):
    """
    Load embeddings from saved .pt files.

    Args:
        embedding_dir: Directory containing saved embeddings
        max_files_per_attack: Maximum number of files to load per attack type
        max_frames_per_file: Maximum number of frames to sample per file

    Returns:
        embeddings: numpy array of shape (N, 1024)
        labels: list of attack type labels
    """
    attack_types = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]

    all_embeddings = []
    all_labels = []

    print("="*80)
    print("Loading Embeddings for t-SNE")
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

        # Limit number of files if specified
        if max_files_per_attack is not None:
            pt_files = pt_files[:max_files_per_attack]

        print(f"\n[INFO] Loading {attack_type}: {len(pt_files)} files")

        attack_embeddings = []

        for pt_file in tqdm(pt_files, desc=f"{attack_type:>10}"):
            try:
                # Load embeddings (T, 1024)
                emb = torch.load(pt_file, map_location='cpu')

                # Sample frames if the file has too many frames
                if emb.shape[0] > max_frames_per_file:
                    # Random sample
                    indices = np.random.choice(emb.shape[0], max_frames_per_file, replace=False)
                    emb = emb[indices]

                # Convert to numpy and add to list
                attack_embeddings.append(emb.numpy())

            except Exception as e:
                print(f"[ERROR] {pt_file.name}: {e}")
                continue

        if len(attack_embeddings) > 0:
            # Concatenate all frames from this attack type
            attack_embeddings = np.vstack(attack_embeddings)  # (N, 1024)
            all_embeddings.append(attack_embeddings)
            all_labels.extend([attack_type] * len(attack_embeddings))

            print(f"[INFO] {attack_type}: {len(attack_embeddings)} frames loaded")

    if len(all_embeddings) == 0:
        raise ValueError("No embeddings loaded!")

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)  # (Total_N, 1024)

    print(f"\n[INFO] Total frames loaded: {len(all_embeddings)}")
    print(f"[INFO] Embedding shape: {all_embeddings.shape}")

    return all_embeddings, all_labels


# =============================================================================
# t-SNE Visualization
# =============================================================================
def plot_tsne(embeddings, labels, output_path, perplexity=30, n_iter=1000,
              random_state=42):
    """
    Apply t-SNE and create visualization.

    Args:
        embeddings: numpy array of shape (N, 1024)
        labels: list of attack type labels
        output_path: path to save the plot
        perplexity: t-SNE perplexity parameter
        n_iter: number of iterations for t-SNE
        random_state: random seed for reproducibility
    """
    print("\n" + "="*80)
    print("Computing t-SNE")
    print("="*80)
    print(f"[INFO] Input shape: {embeddings.shape}")
    print(f"[INFO] Perplexity: {perplexity}")
    print(f"[INFO] Iterations: {n_iter}")

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=random_state, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)

    print(f"[INFO] t-SNE output shape: {embeddings_2d.shape}")

    # Create plot
    print("\n[INFO] Creating plot...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define colors for each attack type
    attack_types = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]
    colors = {
        "bonafide": "#2ecc71",  # Green
        "A01": "#e74c3c",       # Red
        "A02": "#3498db",       # Blue
        "A03": "#f39c12",       # Orange
        "A04": "#9b59b6",       # Purple
        "A05": "#1abc9c",       # Turquoise
        "A06": "#e67e22",       # Dark Orange
    }

    # Plot each attack type
    for attack_type in attack_types:
        # Get indices for this attack type
        indices = [i for i, label in enumerate(labels) if label == attack_type]

        if len(indices) == 0:
            continue

        # Get embeddings for this attack type
        x = embeddings_2d[indices, 0]
        y = embeddings_2d[indices, 1]

        # Plot with transparency
        ax.scatter(x, y, c=colors[attack_type], label=attack_type,
                  alpha=0.6, s=10, edgecolors='none')

    # Customize plot
    ax.set_xlabel("t-SNE Dimension 1", fontsize=14)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=14)
    ax.set_title("t-SNE Visualization of SSL Embeddings", fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12, markerscale=3)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[INFO] Plot saved to: {output_path}")


# =============================================================================
# Plot by Class (Bonafide vs Deepfake)
# =============================================================================
def plot_tsne_binary(embeddings, labels, output_path, perplexity=30, n_iter=1000,
                     random_state=42):
    """
    Apply t-SNE and create binary visualization (bonafide vs deepfake).

    Args:
        embeddings: numpy array of shape (N, 1024)
        labels: list of attack type labels
        output_path: path to save the plot
        perplexity: t-SNE perplexity parameter
        n_iter: number of iterations for t-SNE
        random_state: random seed for reproducibility
    """
    print("\n" + "="*80)
    print("Computing t-SNE (Binary)")
    print("="*80)
    print(f"[INFO] Input shape: {embeddings.shape}")
    print(f"[INFO] Perplexity: {perplexity}")
    print(f"[INFO] Iterations: {n_iter}")

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=random_state, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)

    print(f"[INFO] t-SNE output shape: {embeddings_2d.shape}")

    # Create plot
    print("\n[INFO] Creating binary plot...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define binary labels
    binary_labels = ["bonafide" if label == "bonafide" else "deepfake" for label in labels]

    # Colors
    colors = {
        "bonafide": "#2ecc71",  # Green
        "deepfake": "#e74c3c",  # Red
    }

    # Plot each class
    for class_label in ["bonafide", "deepfake"]:
        # Get indices for this class
        indices = [i for i, label in enumerate(binary_labels) if label == class_label]

        if len(indices) == 0:
            continue

        # Get embeddings for this class
        x = embeddings_2d[indices, 0]
        y = embeddings_2d[indices, 1]

        # Plot with transparency
        ax.scatter(x, y, c=colors[class_label], label=class_label.capitalize(),
                  alpha=0.5, s=10, edgecolors='none')

    # Customize plot
    ax.set_xlabel("t-SNE Dimension 1", fontsize=14)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=14)
    ax.set_title("t-SNE Visualization: Bonafide vs Deepfake", fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=14, markerscale=3)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[INFO] Binary plot saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="t-SNE Visualization of SSL Embeddings"
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
        help='Directory to save plots'
    )
    parser.add_argument(
        '--max_files_per_attack',
        type=int,
        default=100,
        help='Maximum number of files to load per attack type (default: 100)'
    )
    parser.add_argument(
        '--max_frames_per_file',
        type=int,
        default=50,
        help='Maximum number of frames to sample per file (default: 50)'
    )
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity parameter (default: 30)'
    )
    parser.add_argument(
        '--n_iter',
        type=int,
        default=1000,
        help='Number of t-SNE iterations (default: 1000)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("t-SNE Visualization")
    print("="*80)
    print(f"Embedding dir: {args.embedding_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max files per attack: {args.max_files_per_attack}")
    print(f"Max frames per file: {args.max_frames_per_file}")
    print("="*80)

    # Load embeddings
    embeddings, labels = load_embeddings_from_dir(
        embedding_dir=args.embedding_dir,
        max_files_per_attack=args.max_files_per_attack,
        max_frames_per_file=args.max_frames_per_file
    )

    # Plot t-SNE (all attack types)
    output_path = os.path.join(args.output_dir, "tsne_by_attack.png")
    plot_tsne(
        embeddings=embeddings,
        labels=labels,
        output_path=output_path,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=args.random_state
    )

    # Plot t-SNE (binary: bonafide vs deepfake)
    output_path_binary = os.path.join(args.output_dir, "tsne_binary.png")
    plot_tsne_binary(
        embeddings=embeddings,
        labels=labels,
        output_path=output_path_binary,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=args.random_state
    )

    print("\n" + "="*80)
    print("✓ t-SNE Visualization Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
