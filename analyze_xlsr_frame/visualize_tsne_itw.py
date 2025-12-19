"""
t-SNE Visualization of SSL Embeddings for ITW Dataset (Utterance-level)

This script:
1. Loads saved embeddings (T, 1024) for bonafide and spoof
2. Applies mean pooling across time dimension: (T, 1024) → (1024,)
3. Applies t-SNE dimensionality reduction on utterance-level embeddings
4. Plots t-SNE visualization: bonafide vs spoof
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pickle


# =============================================================================
# Load Embeddings (Utterance-level)
# =============================================================================
def load_embeddings_utterance_level(embedding_dir: str, max_files_per_label: int = None):
    """
    Load embeddings from saved .pt files and apply mean pooling.

    Args:
        embedding_dir: Directory containing saved embeddings
        max_files_per_label: Maximum number of files to load per label

    Returns:
        embeddings: numpy array of shape (N, 1024) where N is number of utterances
        labels: list of labels (one per utterance)
    """
    labels_to_load = ["bonafide", "spoof"]

    all_embeddings = []
    all_labels = []

    print("="*80)
    print("Loading Embeddings for t-SNE (Utterance-level) - ITW Dataset")
    print("="*80)

    for label in labels_to_load:
        label_dir = os.path.join(embedding_dir, label)

        if not os.path.exists(label_dir):
            print(f"[WARNING] Missing folder: {label}")
            continue

        pt_files = list(Path(label_dir).glob("*.pt"))

        if len(pt_files) == 0:
            print(f"[WARNING] No embeddings for {label}")
            continue

        # Limit number of files if specified
        if max_files_per_label is not None:
            pt_files = pt_files[:max_files_per_label]

        print(f"\n[INFO] Loading {label}: {len(pt_files)} files")

        label_embeddings = []

        for pt_file in tqdm(pt_files, desc=f"{label:>10}"):
            try:
                # Load embeddings (T, 1024)
                emb = torch.load(pt_file, map_location='cpu')

                # Mean pooling across time dimension: (T, 1024) → (1024,)
                emb_mean = emb.mean(dim=0)  # Average over time axis

                # Convert to numpy and add to list
                label_embeddings.append(emb_mean.numpy())

            except Exception as e:
                print(f"[ERROR] {pt_file.name}: {e}")
                continue

        if len(label_embeddings) > 0:
            # Stack all utterances from this label
            label_embeddings = np.stack(label_embeddings).astype(np.float32)  # (N, 1024)
            all_embeddings.append(label_embeddings)
            all_labels.extend([label] * len(label_embeddings))

            print(f"[INFO] {label}: {len(label_embeddings)} utterances loaded")

    if len(all_embeddings) == 0:
        raise ValueError("No embeddings loaded!")

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings).astype(np.float32)  # (Total_N, 1024)

    print(f"\n[INFO] Total utterances loaded: {len(all_embeddings)}")
    print(f"[INFO] Embedding shape: {all_embeddings.shape}")
    print(f"[INFO] Embedding dtype: {all_embeddings.dtype}")

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
        labels: list of labels
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

    # Standardize features
    print("[INFO] Standardizing features...")
    print(f"[DEBUG] Embeddings dtype before scaling: {embeddings.dtype}")
    print(f"[DEBUG] Embeddings shape before scaling: {embeddings.shape}")

    # Ensure float type
    embeddings = embeddings.astype(np.float32)

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    print(f"[DEBUG] Embeddings_scaled dtype: {embeddings_scaled.dtype}")

    # Apply t-SNE
    print("[INFO] Applying t-SNE...")

    # Calculate learning rate based on sample size
    n_samples = embeddings_scaled.shape[0]
    learning_rate = max(n_samples / 12.0, 50.0)

    print(f"[DEBUG] Learning rate: {learning_rate}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1,
        learning_rate=learning_rate
    )
    embeddings_2d = tsne.fit_transform(embeddings_scaled)

    print(f"[INFO] t-SNE output shape: {embeddings_2d.shape}")

    # Create plot
    print("\n[INFO] Creating plot...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define colors
    colors = {
        "bonafide": "#2ecc71",  # Green
        "spoof": "#e74c3c",     # Red
    }

    # Markers
    markers = {
        "bonafide": "o",  # Circle
        "spoof": "^",     # Triangle
    }

    # Plot each label
    for label in ["bonafide", "spoof"]:
        # Get indices for this label
        indices = [i for i, lbl in enumerate(labels) if lbl == label]

        if len(indices) == 0:
            continue

        # Get embeddings for this label
        x = embeddings_2d[indices, 0]
        y = embeddings_2d[indices, 1]

        # Plot with transparency
        ax.scatter(x, y, c=colors[label], label=label.capitalize(),
                  alpha=0.6, s=30, edgecolors='none', marker=markers[label])

    # Customize plot
    ax.set_xlabel("t-SNE Dimension 1", fontsize=14)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=14)
    ax.set_title("t-SNE Visualization: Bonafide vs Spoof (ITW Dataset)", fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=14, markerscale=2)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[INFO] Plot saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="t-SNE Visualization of SSL Embeddings for ITW Dataset (Utterance-level)"
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
        default='./results_itw/',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--max_files_per_label',
        type=int,
        default=None,
        help='Maximum number of files to load per label (default: None = all files)'
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
    print("t-SNE Visualization (Utterance-level) - ITW Dataset")
    print("="*80)
    print(f"Embedding dir: {args.embedding_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max files per label: {args.max_files_per_label if args.max_files_per_label else 'All'}")
    print("="*80)

    # Load embeddings (utterance-level)
    embeddings, labels = load_embeddings_utterance_level(
        embedding_dir=args.embedding_dir,
        max_files_per_label=args.max_files_per_label
    )

    # Plot t-SNE
    output_path = os.path.join(args.output_dir, "tsne_itw_utterance.png")
    plot_tsne(
        embeddings=embeddings,
        labels=labels,
        output_path=output_path,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=args.random_state
    )

    print("\n" + "="*80)
    print("✓ t-SNE Visualization Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
