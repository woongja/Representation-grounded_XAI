"""
Extract and Save SSL Embeddings for ITW Dataset (Vanilla XLSR)

This script extracts frame-level embeddings from the vanilla wav2vec2-XLSR model
and saves them to disk for ITW dataset.
"""

import os
import sys
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from tqdm import tqdm
import argparse
import pickle
import fairseq

from itw_protocol_parser import parse_itw_protocol


# =============================================================================
# SSL Model (Vanilla wav2vec2-XLSR)
# =============================================================================
class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        cp_path = '/home/woongjae/wildspoof/xlsr2_300m.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        # Put the model to GPU if it's not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        # Input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb


# =============================================================================
# Audio Loading Functions
# =============================================================================
def load_audio(file_path: str, target_sr: int = 16000, max_length: int = None) -> torch.Tensor:
    """
    Load audio file and resample if needed.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 16000 Hz)
        max_length: Maximum length in samples (optional)

    Returns:
        audio: Audio waveform tensor of shape (length,)
    """
    audio, sr = torchaudio.load(file_path)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)

    # Remove channel dimension
    audio = audio.squeeze(0)

    # Truncate if needed
    if max_length is not None and audio.shape[0] > max_length:
        audio = audio[:max_length]

    return audio


# =============================================================================
# Embedding Extraction and Saving
# =============================================================================
def extract_and_save_embeddings(
    label_dict: dict,
    output_dir: str,
    device: str = 'cuda',
    resume: bool = True
):
    """
    Extract SSL embeddings using vanilla wav2vec2-XLSR model.

    Args:
        label_dict: Dictionary mapping labels to file paths
        output_dir: Directory to save embeddings
        device: Device to run model on ('cuda' or 'cpu')
        resume: If True, skip files that already have saved embeddings
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize vanilla XLSR model
    print("="*80)
    print("Initializing Vanilla wav2vec2-XLSR Model")
    print("="*80)

    ssl_model = SSLModel(device)
    ssl_model.eval()

    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Resume mode: {resume}")
    print(f"[INFO] Output dimension: {ssl_model.out_dim}")

    # Track saved files
    embedding_index = {}

    # Process each label
    for label in sorted(label_dict.keys()):
        file_list = label_dict[label]

        print("\n" + "="*80)
        print(f"Processing {label}: {len(file_list)} files")
        print("="*80)

        # Create label-specific directory
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)

        # Filter files if resuming
        files_to_process = []
        for file_path in file_list:
            filename = Path(file_path).stem  # e.g., "0", "1234"
            save_path = os.path.join(save_dir, f"{filename}.pt")

            if resume and os.path.exists(save_path):
                continue  # Skip already processed files

            files_to_process.append(file_path)

        if resume and len(files_to_process) < len(file_list):
            print(f"[INFO] Resuming: {len(file_list) - len(files_to_process)} files already processed")
            print(f"[INFO] Processing remaining {len(files_to_process)} files")

        if len(files_to_process) == 0:
            print(f"[INFO] All files already processed, skipping...")
            embedding_index[label] = [Path(f).stem for f in file_list]
            continue

        # Process one by one
        saved_files = []
        shape_printed = False  # Flag to print shape info only once

        for file_path in tqdm(files_to_process, desc=f"{label:>10}"):
            try:
                # Load single audio file
                audio = load_audio(file_path)  # (length,)
                audio = audio.unsqueeze(0).to(device)  # (1, length)

                # Extract embeddings
                with torch.no_grad():
                    embedding = ssl_model.extract_feat(audio)  # (1, T, 1024)
                    embedding = embedding.squeeze(0)  # (T, 1024)

                # Print shape info for first file
                if not shape_printed:
                    print(f"\n[INFO] Embedding shape info:")
                    print(f"  Input audio shape:  {audio.shape}")
                    print(f"  Output emb shape:   {embedding.shape}")
                    print(f"  Saved emb shape:    (T, 1024) where T={embedding.shape[0]}")
                    print(f"  Model: Vanilla wav2vec2-XLSR")
                    shape_printed = True

                # Save embedding
                filename = Path(file_path).stem
                save_path = os.path.join(save_dir, f"{filename}.pt")
                torch.save(embedding.cpu(), save_path)
                saved_files.append(filename)

            except Exception as e:
                print(f"\n[ERROR] Failed to process {Path(file_path).name}: {e}")
                continue

        # Update index
        all_saved = list(Path(save_dir).glob("*.pt"))
        embedding_index[label] = [p.stem for p in all_saved]

        print(f"\n[INFO] Saved {len(saved_files)} new embeddings")
        print(f"[INFO] Total embeddings for {label}: {len(embedding_index[label])}")

    # Save embedding index
    index_path = os.path.join(output_dir, "embedding_index.pkl")
    with open(index_path, 'wb') as f:
        pickle.dump(embedding_index, f)

    print("\n" + "="*80)
    print("Embedding Extraction Complete!")
    print("="*80)
    print(f"[INFO] Embedding index saved to: {index_path}")

    # Print summary statistics
    print("\n[INFO] Summary:")
    total_embeddings = 0
    for label in sorted(embedding_index.keys()):
        count = len(embedding_index[label])
        total_embeddings += count
        print(f"  {label:>10}: {count:>6} embeddings")
    print(f"  {'TOTAL':>10}: {total_embeddings:>6} embeddings")

    # Estimate disk usage
    sample_file = None
    for label in embedding_index.keys():
        files = list(Path(os.path.join(output_dir, label)).glob("*.pt"))
        if files:
            sample_file = files[0]
            break

    if sample_file:
        sample_size = os.path.getsize(sample_file)
        estimated_total = sample_size * total_embeddings / (1024**3)  # GB
        print(f"\n[INFO] Estimated disk usage: {estimated_total:.2f} GB")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract and save SSL embeddings for ITW dataset using vanilla XLSR"
    )
    parser.add_argument(
        '--protocol_path',
        type=str,
        default='/home/woongjae/ADD_LAB/Wav-Spec_ADD/protocols/protocol_itw.txt',
        help='Path to ITW protocol file'
    )
    parser.add_argument(
        '--audio_base_dir',
        type=str,
        default='/home/woongjae/ADD_LAB/Datasets/itw',
        help='Base directory containing .wav audio files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/nvme3/wj/embeddings_itw/',
        help='Directory to save embeddings (default: /nvme3/wj/embeddings_itw/)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='If set, reprocess all files (default: resume from existing)'
    )

    args = parser.parse_args()

    print("="*80)
    print("ITW Dataset - Vanilla XLSR Embedding Extraction")
    print("="*80)
    print(f"Protocol: {args.protocol_path}")
    print(f"Audio base dir: {args.audio_base_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*80)

    # Parse protocol
    label_dict = parse_itw_protocol(
        protocol_path=args.protocol_path,
        base_dir=args.audio_base_dir
    )

    # Extract and save embeddings
    extract_and_save_embeddings(
        label_dict=label_dict,
        output_dir=args.output_dir,
        device=args.device,
        resume=not args.no_resume
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
