"""
Extract and Save SSL Embeddings

This script extracts frame-level embeddings from audio files using wav2vec2-XLSR
and saves them to disk for later analysis.

This is a one-time preprocessing step that significantly speeds up subsequent analyses.
"""

import os
import sys
import torch
import torch.nn as nn
import torchaudio
import fairseq
from pathlib import Path
from tqdm import tqdm
import argparse
import pickle

from asvspoof_protocol_parser import parse_asvspoof_protocol


# =============================================================================
# SSL Model Definition (from conformertcm.py)
# =============================================================================

class SSLModel(nn.Module):
    """
    SSL Model using wav2vec2-XLSR.

    This class is copied from model/conformertcm.py to make the script self-contained.
    """
    def __init__(self, device):
        super(SSLModel, self).__init__()
        cp_path = '/home/woongjae/wildspoof/xlsr2_300m.pt'   # Pre-trained XLSR model path
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        """
        Extract frame-level features from audio.

        Args:
            input_data: Audio tensor of shape (batch, length) or (batch, length, 1)

        Returns:
            emb: Frame-level embeddings of shape (batch, T, 1024)
        """
        # Put the model to GPU if it's not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.eval()

        # Input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb


def load_audio(file_path: str, target_sr: int = 16000, max_length: int = None) -> torch.Tensor:
    """
    Load audio file and resample if needed.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 16000 Hz)
        max_length: Maximum length in samples (optional, for truncation)

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


def load_audio_batch(file_paths: list, target_sr: int = 16000) -> torch.Tensor:
    """
    Load multiple audio files and pad to same length.

    Args:
        file_paths: List of audio file paths
        target_sr: Target sample rate

    Returns:
        batch_audio: Tensor of shape (batch_size, max_length)
    """
    audios = []
    max_length = 0

    # Load all audio files
    for file_path in file_paths:
        audio = load_audio(file_path, target_sr=target_sr)
        audios.append(audio)
        max_length = max(max_length, audio.shape[0])

    # Pad to same length
    padded_audios = []
    for audio in audios:
        if audio.shape[0] < max_length:
            padding = torch.zeros(max_length - audio.shape[0])
            audio = torch.cat([audio, padding])
        padded_audios.append(audio)

    batch_audio = torch.stack(padded_audios)
    return batch_audio


def extract_and_save_embeddings(
    attack_dict: dict,
    output_dir: str,
    device: str = 'cuda',
    batch_size: int = 32,
    resume: bool = True
):
    """
    Extract SSL embeddings for all audio files and save to disk.

    Args:
        attack_dict: Dictionary mapping attack types to file paths
                    {"bonafide": [paths], "A01": [paths], ...}
        output_dir: Directory to save embeddings (e.g., /nvme3/wj/embeddings/)
        device: Device to run model on ('cuda' or 'cpu')
        batch_size: Batch size for processing
        resume: If True, skip files that already have saved embeddings

    Saves:
        output_dir/bonafide/LA_T_xxx.pt
        output_dir/A01/LA_T_xxx.pt
        ...
        output_dir/embedding_index.pkl
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize SSL model
    print("="*80)
    print("Initializing SSL Model (wav2vec2-XLSR)")
    print("="*80)

    ssl_model = SSLModel(device=device)
    ssl_model.model.eval()

    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Resume mode: {resume}")

    # Track saved files
    embedding_index = {}

    # Process each attack type
    for attack_type in sorted(attack_dict.keys()):
        file_list = attack_dict[attack_type]

        print("\n" + "="*80)
        print(f"Processing {attack_type}: {len(file_list)} files")
        print("="*80)

        # Create attack-specific directory
        save_dir = os.path.join(output_dir, attack_type)
        os.makedirs(save_dir, exist_ok=True)

        # Filter files if resuming
        files_to_process = []
        for file_path in file_list:
            filename = Path(file_path).stem  # LA_T_xxx
            save_path = os.path.join(save_dir, f"{filename}.pt")

            if resume and os.path.exists(save_path):
                continue  # Skip already processed files

            files_to_process.append(file_path)

        if resume and len(files_to_process) < len(file_list):
            print(f"[INFO] Resuming: {len(file_list) - len(files_to_process)} files already processed")
            print(f"[INFO] Processing remaining {len(files_to_process)} files")

        if len(files_to_process) == 0:
            print(f"[INFO] All files already processed, skipping...")
            embedding_index[attack_type] = [Path(f).stem for f in file_list]
            continue

        # Process in batches
        saved_files = []

        for i in tqdm(range(0, len(files_to_process), batch_size), desc=f"{attack_type}"):
            batch_files = files_to_process[i:i+batch_size]

            try:
                # Load audio batch
                batch_audio = load_audio_batch(batch_files)  # (B, max_length)
                batch_audio = batch_audio.to(device)

                # Extract embeddings
                with torch.no_grad():
                    embeddings = ssl_model.extract_feat(batch_audio)  # (B, T, 1024)

                # Save individual embeddings
                for j, file_path in enumerate(batch_files):
                    filename = Path(file_path).stem  # LA_T_xxx
                    save_path = os.path.join(save_dir, f"{filename}.pt")

                    # Save embedding for this file (T, 1024)
                    torch.save(embeddings[j].cpu(), save_path)
                    saved_files.append(filename)

            except Exception as e:
                print(f"\n[ERROR] Failed to process batch starting at index {i}: {e}")
                print(f"[ERROR] Files in failed batch: {[Path(f).name for f in batch_files]}")
                continue

        # Update index
        all_saved = list(Path(save_dir).glob("*.pt"))
        embedding_index[attack_type] = [p.stem for p in all_saved]

        print(f"[INFO] Saved {len(saved_files)} new embeddings")
        print(f"[INFO] Total embeddings for {attack_type}: {len(embedding_index[attack_type])}")

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
    for attack_type in sorted(embedding_index.keys()):
        count = len(embedding_index[attack_type])
        total_embeddings += count
        print(f"  {attack_type:>10}: {count:>6} embeddings")
    print(f"  {'TOTAL':>10}: {total_embeddings:>6} embeddings")

    # Estimate disk usage
    sample_file = None
    for attack_type in embedding_index.keys():
        files = list(Path(os.path.join(output_dir, attack_type)).glob("*.pt"))
        if files:
            sample_file = files[0]
            break

    if sample_file:
        sample_size = os.path.getsize(sample_file)
        estimated_total = sample_size * total_embeddings / (1024**3)  # GB
        print(f"\n[INFO] Estimated disk usage: {estimated_total:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save SSL embeddings from ASVspoof audio files"
    )
    parser.add_argument(
        '--protocol_path',
        type=str,
        required=True,
        help='Path to ASVspoof protocol file'
    )
    parser.add_argument(
        '--audio_base_dir',
        type=str,
        required=True,
        help='Base directory containing .flac audio files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/nvme3/wj/embeddings/',
        help='Directory to save embeddings (default: /nvme3/wj/embeddings/)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for processing (default: 16)'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='If set, reprocess all files (default: resume from existing)'
    )

    args = parser.parse_args()

    print("="*80)
    print("SSL Embedding Extraction")
    print("="*80)
    print(f"Protocol: {args.protocol_path}")
    print(f"Audio base dir: {args.audio_base_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("="*80)

    # Parse protocol
    attack_dict = parse_asvspoof_protocol(
        protocol_path=args.protocol_path,
        base_dir=args.audio_base_dir
    )

    # Extract and save embeddings
    extract_and_save_embeddings(
        attack_dict=attack_dict,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        resume=not args.no_resume
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
