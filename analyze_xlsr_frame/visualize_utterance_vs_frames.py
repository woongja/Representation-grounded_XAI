"""
Utterance vs Frames t-SNE Visualization

This script visualizes how close individual frames are to their utterance-level representation
in t-SNE space. For each attack type, it:
1. Randomly selects one utterance
2. Computes the utterance-level embedding (mean pooling)
3. Takes all frames from that utterance
4. Applies t-SNE to visualize utterance (as a star) and frames (as dots)
"""

import os
import torch
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import random
import torchaudio
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Visualize Waveform with Distance Mapping
# =============================================================================
def visualize_waveform_with_distances(audio_path: str, frame_distances: np.ndarray,
                                      attack_type: str, output_path: str):
    """
    Visualize waveform with color-coded frame distances.

    Args:
        audio_path: Path to audio file
        frame_distances: Array of distances for each frame (T,)
        attack_type: Attack type label
        output_path: Path to save the plot
    """
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    audio = audio.squeeze().numpy()  # Convert to 1D numpy array

    # Calculate frame boundaries (assuming 320 hop size for 16kHz audio)
    # wav2vec2 typically uses 20ms frames with 10ms stride
    hop_length = 160  # 10ms at 16kHz
    frame_length = 400  # 25ms at 16kHz

    n_frames = len(frame_distances)

    # Create time axis
    time = np.arange(len(audio)) / sr

    # Normalize distances to [0, 1] for colormap
    dist_min = frame_distances.min()
    dist_max = frame_distances.max()
    distances_normalized = (frame_distances - dist_min) / (dist_max - dist_min + 1e-8)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Create colormap: blue (far) to red (close)
    # Reverse it so close = red, far = blue
    cmap = plt.cm.coolwarm_r

    # Plot 1: Waveform with colored segments
    for i in range(n_frames):
        start_sample = i * hop_length
        end_sample = min(start_sample + frame_length, len(audio))

        if start_sample >= len(audio):
            break

        color = cmap(distances_normalized[i])
        ax1.plot(time[start_sample:end_sample], audio[start_sample:end_sample],
                color=color, linewidth=0.5, alpha=0.7)

    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Amplitude", fontsize=11)
    ax1.set_title(f"Waveform colored by Frame-to-Utterance Distance: {attack_type}\n"
                 f"Red = Close to utterance | Blue = Far from utterance",
                 fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')

    # Plot 2: Distance over time
    frame_times = np.arange(n_frames) * hop_length / sr
    scatter = ax2.scatter(frame_times, frame_distances, c=distances_normalized,
                         cmap=cmap, s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.plot(frame_times, frame_distances, color='gray', alpha=0.3, linewidth=1)

    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Distance to Utterance", fontsize=11)
    ax2.set_title("Frame Distance over Time", fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Normalized Distance\n(0=Close, 1=Far)", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Waveform plot saved: {output_path}")


# =============================================================================
# Visualize Single Utterance vs Frames
# =============================================================================
def visualize_utterance_vs_frames(embedding_path: str, audio_path: str,
                                   attack_type: str, output_path: str,
                                   waveform_output_path: str,
                                   perplexity=30, n_iter=1000, random_state=42):
    """
    Visualize a single utterance and its frames in t-SNE space.

    Args:
        embedding_path: Path to the embedding file (.pt)
        attack_type: Attack type label
        output_path: Path to save the plot
        perplexity: t-SNE perplexity
        n_iter: t-SNE iterations
        random_state: Random seed
    """
    # Load embeddings
    emb = torch.load(embedding_path, map_location='cpu')  # (T, 1024)

    print(f"\n[INFO] Processing {attack_type}")
    print(f"  File: {Path(embedding_path).name}")
    print(f"  Shape: {emb.shape}")

    # Compute utterance-level embedding (mean pooling)
    utterance_emb = emb.mean(dim=0, keepdim=True)  # (1, 1024)

    # Calculate distance: || h_utt - mean(h_frame(t)) ||
    # Note: utterance_emb is already mean(h_frame(t)), so distance should be 0
    # But let's compute mean of frames again to verify
    frames_mean = emb.mean(dim=0, keepdim=True)  # (1, 1024)
    distance = torch.norm(utterance_emb - frames_mean, p=2).item()

    print(f"  Distance || h_utt - mean(h_frame) ||: {distance:.6f}")

    # Also compute average distance from each frame to utterance
    frame_distances = torch.norm(emb - utterance_emb, p=2, dim=1)  # (T,)
    avg_frame_distance = frame_distances.mean().item()
    std_frame_distance = frame_distances.std().item()

    print(f"  Avg distance from frames to utterance: {avg_frame_distance:.4f} ± {std_frame_distance:.4f}")

    # Combine utterance and frames
    # First row: utterance, rest: frames
    all_emb = torch.cat([utterance_emb, emb], dim=0)  # (1+T, 1024)
    all_emb_np = all_emb.numpy().astype(np.float32)

    print(f"  Combined shape: {all_emb_np.shape}")
    print(f"  Utterance: 1, Frames: {emb.shape[0]}")

    # Standardize
    scaler = StandardScaler()
    all_emb_scaled = scaler.fit_transform(all_emb_np)

    # Apply t-SNE
    print(f"  Running t-SNE...")
    n_samples = all_emb_scaled.shape[0]
    learning_rate = max(n_samples / 12.0, 50.0)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, n_samples - 1),
        n_iter=n_iter,
        random_state=random_state,
        verbose=0,
        learning_rate=learning_rate
    )
    embeddings_2d = tsne.fit_transform(all_emb_scaled)

    # Separate utterance and frames
    utterance_2d = embeddings_2d[0]  # First point
    frames_2d = embeddings_2d[1:]    # Rest

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot frames as small dots
    ax.scatter(frames_2d[:, 0], frames_2d[:, 1],
              c='#3498db', alpha=0.5, s=30,
              edgecolors='none', label='Frames')

    # Plot utterance as a star
    ax.scatter(utterance_2d[0], utterance_2d[1],
              c='#e74c3c', marker='*', s=500,
              edgecolors='black', linewidths=2,
              label='Utterance (mean)', zorder=10)

    # Customize plot
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    title_text = f"Utterance vs Frames: {attack_type}\n"
    title_text += f"({emb.shape[0]} frames) | Avg Dist: {avg_frame_distance:.4f} ± {std_frame_distance:.4f}"
    ax.set_title(title_text, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved: {output_path}")

    # Visualize waveform with distance mapping
    visualize_waveform_with_distances(
        audio_path=audio_path,
        frame_distances=frame_distances.numpy(),
        attack_type=attack_type,
        output_path=waveform_output_path
    )

    return {
        "attack_type": attack_type,
        "n_frames": emb.shape[0],
        "avg_distance": avg_frame_distance,
        "std_distance": std_frame_distance
    }


# =============================================================================
# Process All Attack Types
# =============================================================================
def process_all_attack_types(embedding_dir: str, audio_base_dir: str,
                             output_dir: str, perplexity=30, n_iter=1000,
                             random_state=42):
    """
    Process all attack types and create visualizations.

    Args:
        embedding_dir: Directory containing embeddings
        audio_base_dir: Base directory containing audio files
        output_dir: Directory to save plots
        perplexity: t-SNE perplexity
        n_iter: t-SNE iterations
        random_state: Random seed
    """
    attack_types = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]

    print("="*80)
    print("Utterance vs Frames t-SNE Visualization")
    print("="*80)
    print(f"Embedding dir: {embedding_dir}")
    print(f"Output dir: {output_dir}")
    print("="*80)

    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)

    os.makedirs(output_dir, exist_ok=True)

    # Store statistics
    statistics = []

    for attack_type in attack_types:
        attack_dir = os.path.join(embedding_dir, attack_type)

        if not os.path.exists(attack_dir):
            print(f"\n[WARNING] Missing folder: {attack_type}")
            continue

        # Get all .pt files
        pt_files = list(Path(attack_dir).glob("*.pt"))

        if len(pt_files) == 0:
            print(f"\n[WARNING] No embeddings for {attack_type}")
            continue

        # Randomly select one file
        selected_file = random.choice(pt_files)

        # Find corresponding audio file
        file_stem = selected_file.stem  # e.g., "LA_T_1234567"
        audio_file = Path(audio_base_dir) / f"{file_stem}.flac"

        if not audio_file.exists():
            print(f"\n[WARNING] Audio file not found: {audio_file}")
            continue

        # Output paths
        output_path = os.path.join(output_dir, f"utterance_vs_frames_{attack_type}.png")
        waveform_output_path = os.path.join(output_dir, f"waveform_distance_{attack_type}.png")

        # Visualize
        stats = visualize_utterance_vs_frames(
            embedding_path=str(selected_file),
            audio_path=str(audio_file),
            attack_type=attack_type,
            output_path=output_path,
            waveform_output_path=waveform_output_path,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state
        )
        statistics.append(stats)

    # Print summary statistics
    print("\n" + "="*80)
    print("Distance Statistics Summary")
    print("="*80)
    print(f"{'Attack':<10} {'N Frames':>10} {'Avg Distance':>15} {'Std Distance':>15}")
    print("-"*80)
    for stat in statistics:
        print(f"{stat['attack_type']:<10} {stat['n_frames']:>10} "
              f"{stat['avg_distance']:>15.4f} {stat['std_distance']:>15.4f}")
    print("="*80)

    print("\n" + "="*80)
    print("✓ All visualizations complete!")
    print("="*80)
    print(f"Plots saved to: {output_dir}")
    print("  - utterance_vs_frames_bonafide.png")
    print("  - utterance_vs_frames_A01.png")
    print("  - utterance_vs_frames_A02.png")
    print("  - utterance_vs_frames_A03.png")
    print("  - utterance_vs_frames_A04.png")
    print("  - utterance_vs_frames_A05.png")
    print("  - utterance_vs_frames_A06.png")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize utterance vs frames in t-SNE space"
    )
    parser.add_argument(
        '--embedding_dir',
        type=str,
        required=True,
        help='Directory containing saved embeddings'
    )
    parser.add_argument(
        '--audio_base_dir',
        type=str,
        required=True,
        help='Base directory containing audio files (.flac)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/utterance_vs_frames/',
        help='Directory to save plots'
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

    process_all_attack_types(
        embedding_dir=args.embedding_dir,
        audio_base_dir=args.audio_base_dir,
        output_dir=args.output_dir,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
