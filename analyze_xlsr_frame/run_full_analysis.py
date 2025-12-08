"""
Complete Analysis Pipeline

This script runs the complete analysis pipeline for the paper:
"Frame-level Temporal Difference Learning for Partial Deepfake Speech Detection"

Pipeline:
    1. Parse ASVspoof protocol file
    2. Extract SSL embeddings (optional, can skip if already done)
    3. Run Figure 2 analysis (cosine similarity with Eq 1 & 2)
"""

import os
import sys
import argparse
import time

from asvspoof_protocol_parser import parse_asvspoof_protocol
from extract_embeddings import extract_and_save_embeddings
from figure2_cosine_similarity_analysis import (
    compute_cosine_similarity_stats_from_embeddings,
    plot_figure2_boxplots,
    save_results as save_fig2_results,
    print_statistics as print_fig2_stats
)


def main():
    parser = argparse.ArgumentParser(
        description="Complete Analysis Pipeline for Temporal Difference Learning"
    )

    # Protocol and audio
    parser.add_argument(
        '--protocol_path',
        type=str,
        required=True,
        help='Path to ASVspoof protocol file'
    )
    parser.add_argument(
        '--audio_base_dir',
        type=str,
        default=None,
        help='Base directory containing .flac audio files (required for embedding extraction)'
    )

    # Embedding extraction
    parser.add_argument(
        '--extract_embeddings',
        action='store_true',
        help='Extract embeddings (skip if already done)'
    )
    parser.add_argument(
        '--embedding_dir',
        type=str,
        default='/nvme3/wj/embeddings/',
        help='Directory to save/load embeddings'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for embedding extraction (cuda or cpu)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for embedding extraction'
    )

    # Analysis options
    parser.add_argument(
        '--skip_figure2',
        action='store_true',
        help='Skip Figure 2 analysis'
    )

    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/',
        help='Directory to save analysis results'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("COMPLETE ANALYSIS PIPELINE")
    print("Frame-level Temporal Difference Learning for Partial Deepfake Detection")
    print("="*80)
    print(f"Protocol file: {args.protocol_path}")
    print(f"Embedding directory: {args.embedding_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    start_time_total = time.time()

    # ========================================================================
    # Step 1: Parse Protocol (always run to show stats)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Parsing Protocol File")
    print("="*80)

    attack_dict = parse_asvspoof_protocol(
        protocol_path=args.protocol_path,
        base_dir=args.audio_base_dir if args.extract_embeddings else None
    )

    # ========================================================================
    # Step 2: Extract Embeddings (optional)
    # ========================================================================
    if args.extract_embeddings:
        print("\n" + "="*80)
        print("STEP 2: Extracting SSL Embeddings")
        print("="*80)

        if args.audio_base_dir is None:
            print("[ERROR] --audio_base_dir is required for embedding extraction")
            sys.exit(1)

        start_time = time.time()

        extract_and_save_embeddings(
            attack_dict=attack_dict,
            output_dir=args.embedding_dir,
            device=args.device,
            batch_size=args.batch_size,
            resume=True
        )

        elapsed = time.time() - start_time
        print(f"\n[INFO] Embedding extraction completed in {elapsed/60:.2f} minutes")
    else:
        print("\n" + "="*80)
        print("STEP 2: Skipping Embedding Extraction (--extract_embeddings not set)")
        print("="*80)
        print(f"[INFO] Using existing embeddings from: {args.embedding_dir}")

    # ========================================================================
    # Step 3: Figure 2 Analysis
    # ========================================================================
    if not args.skip_figure2:
        print("\n" + "="*80)
        print("STEP 3: Figure 2 Analysis (Cosine Similarity - Eq 1 & 2)")
        print("="*80)

        start_time = time.time()

        # Compute
        mean_results, std_results = compute_cosine_similarity_stats_from_embeddings(
            args.embedding_dir
        )

        # Print statistics
        print_fig2_stats(mean_results, std_results)

        # Save results
        fig2_result_path = os.path.join(args.output_dir, "figure2_results.pkl")
        save_fig2_results(mean_results, std_results, fig2_result_path)

        # Plot
        plot_figure2_boxplots(mean_results, std_results, args.output_dir)

        elapsed = time.time() - start_time
        print(f"\n[INFO] Figure 2 analysis completed in {elapsed:.2f} seconds")
    else:
        print("\n" + "="*80)
        print("STEP 3: Skipping Figure 2 Analysis (--skip_figure2 set)")
        print("="*80)

    # ========================================================================
    # Summary
    # ========================================================================
    total_elapsed = time.time() - start_time_total

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"Total time: {total_elapsed/60:.2f} minutes")
    print(f"\nResults saved to: {args.output_dir}")
    print("\nGenerated files:")

    if not args.skip_figure2:
        print("  - figure2_results.pkl")
        print("  - figure2_cosine_similarity_boxplots.png")
        print("  - figure2_cosine_similarity_boxplots.pdf")
        print("  - figure2_mean_boxplot.png")
        print("  - figure2_std_boxplot.png")

    print("\n" + "="*80)
    print("✓ All Done!")
    print("="*80)


if __name__ == "__main__":
    main()
