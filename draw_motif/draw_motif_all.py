#!/usr/bin/env python3
"""
draw_motif_all.py

Multi-process motif drawing script for all RNA modifications.

This script runs the complete motif extraction and visualization pipeline
for multiple RNA modifications in parallel using multiprocessing.

Usage:
    python draw_motif_all.py --data-type train --num-processes 3
    python draw_motif_all.py --data-type train --num-processes 3 --sample-size 5000 --seed 666

Requirements:
    - CUDA libraries must be available in LD_LIBRARY_PATH
    - TreeX model weights file
    - MultiRM HDF5 data file
    - R script for sequence alignment (alignment_local.R)
    - MEME Suite tools (STREME, Tomtom)
"""

import os
import sys
import argparse
import multiprocessing as mp
from functools import partial
import traceback
import logging
from datetime import datetime

# Set up CUDA library path before importing torch
os.environ['LD_LIBRARY_PATH'] = os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import numpy as np
import pandas as pd

# Add draw_motif directory to path
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    run_pipeline_part1, run_pipeline_part2,
    draw_motif_logos, display_tomtom_results
)


# Global log file path (shared across processes)
_log_file_path = None


def setup_logging(log_dir='draw_motif/draw_motif/logs', is_child_process=False):
    """
    Set up logging configuration with stdout/stderr capture.

    Args:
        log_dir: Directory to save log files
        is_child_process: If True, use existing log file instead of creating new one

    Returns:
        logger object and original stdout/stderr
    """
    global _log_file_path

    os.makedirs(log_dir, exist_ok=True)

    if is_child_process or _log_file_path is not None:
        # Child process uses parent's log file
        log_file = _log_file_path
    else:
        # Main process creates new log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'motif_pipeline_{timestamp}.log')
        _log_file_path = log_file

    # Create a custom handler that writes to both file and stdout
    class StreamCapture:
        def __init__(self, original_stream, log_file):
            self.original_stream = original_stream
            self.log_file = log_file

        def write(self, data):
            # Write to original stream (console)
            self.original_stream.write(data)
            self.original_stream.flush()
            # Write to log file
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(data)
                    f.flush()
            except Exception:
                pass  # Avoid errors during logging

        def flush(self):
            self.original_stream.flush()

    # Configure logging for structured log messages
    if not is_child_process:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
    else:
        logger = logging.getLogger(__name__)

    # Capture stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = StreamCapture(original_stdout, log_file)
    sys.stderr = StreamCapture(original_stderr, log_file)

    return logger, original_stdout, original_stderr


# Global logger and original streams
logger = None
original_stdout = None
original_stderr = None


# Configuration


# Configuration
# All available RNA modifications (for validation)
ALL_MODIFICATIONS = ['hAm', 'hCm', 'hGm', 'hUm', 'hm1A', 'hm5C', 'hm5U', 'hm6A', 'hm6Am', 'hm7G', 'hPsi', 'Atol']

# Default eps values for each modification
# The modifications to process are determined by this dictionary's keys
# Example: To only process hAm and hm6A, set: DEFAULT_EPS_DICT = {'hAm': 0.3, 'hm6A': 0.3}
# These values are also used as recommended eps when --auto-eps is enabled
DEFAULT_EPS_DICT = {
    'hAm': 0.2,      # hA modifications - medium diversity
    'hCm': 0.2,      # hC modifications - medium diversity
    'hGm': 0.3,      # hG modifications - medium diversity
    'hUm': 0.3,      # hU modifications - medium diversity
    'hm1A': 0.3,     # m1A - higher diversity
    'hm5C': 0.3,     # m5C - higher diversity
    'hm5U': 0.3,     # m5U - higher diversity
    'hm6A': 0.2,     # m6A - very diverse motifs
    'hm6Am': 0.3,    # m6Am - very diverse motifs
    'hm7G': 0.3,     # m7G - higher diversity
    'hPsi': 0.3,     # Pseudouridine - medium diversity
    'Atol': 0.2      # Atol - lower diversity
}

# Fixed parameters
LENGTH = 51
NUM_TASK = 12


def process_single_modification(RM, config, eps_dict=None, png_dir=None, pdf_dir=None, tomtom_dir=None, dna=False):
    """
    Process a single RNA modification type and save results immediately.

    Args:
        RM: RNA modification type name
        config: Configuration dictionary containing:
            - data_path: Path to HDF5 data file
            - model_weights: Path to model weights
            - data_type: 'train', 'valid', or 'test'
            - w: Window size for IG aggregation
            - k: Top-K windows to select
            - sample_size: Number of samples to process
            - seed: Random seed for sample selection
            - alignment_script: Path to alignment script
            - batch_size: Batch size for model inference (default: 128)
        eps_dict: Dictionary mapping modification names to eps values
        png_dir: Directory to save PNG files
        pdf_dir: Directory to save PDF files
        tomtom_dir: Directory to save Tomtom result TXT files
        dna: If True, pass --dna flag to STREME (treats sequences as DNA with ACGT alphabet)

    Returns:
        Dictionary containing results or error information
    """
    if eps_dict is None:
        eps_dict = DEFAULT_EPS_DICT

    print(f"\n{'='*80}")
    print(f"Processing modification: {RM}")
    print(f"{'='*80}")

    # Get RM index
    RM_index = ALL_MODIFICATIONS.index(RM)

    # Get eps for this modification
    eps = eps_dict.get(RM, 0.3)

    # Set up base directory
    BASE_DIR = f"draw_motif/{config['data_type']}_{RM}_w{config['w']}_k{config['k']}/"
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(f"{BASE_DIR}streme_out", exist_ok=True)
    os.makedirs(f"{BASE_DIR}tomtom_out", exist_ok=True)

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"eps: {eps}")

    result = {
        'RM': RM,
        'eps': eps,
        'BASE_DIR': BASE_DIR
    }

    try:
        # Part 1: Run pipeline up to eps search
        print(f"\n--- Running Part 1 for {RM} ---")
        aligned_short_seqs, scores, best_eps, cluster_analysis_df = run_pipeline_part1(
            data_path=config['data_path'],
            model_weights=config['model_weights'],
            data_type=config['data_type'],
            RM=RM,
            RM_index=RM_index,
            length=LENGTH,
            w=config['w'],
            k=config['k'],
            auto_search_eps=False,  # Use specified eps
            eps_search_range=[0.1, 3],
            eps_step=0.1,
            sample_size=config['sample_size'],
            BASE_DIR=BASE_DIR,
            alignment_script=config['alignment_script'],
            eps=eps,
            seed=config['seed'],  # Random seed for sample selection
            batch_size=config.get('batch_size', 128),  # Batch size for inference
            dna=dna  # Pass --dna flag to STREME
        )

        print(f"\n--- Part 1 completed for {RM} ---")
        print(f"Cluster Analysis Table:")
        print(cluster_analysis_df.to_string(index=False))

        # Part 2: Run pipeline from consensus motifs to validation
        print(f"\n--- Running Part 2 for {RM} ---")
        consensus_motif, ig_scores, pwm_list, used_eps = run_pipeline_part2(
            aligned_short_seqs=aligned_short_seqs,
            scores=scores,
            eps=eps,
            BASE_DIR=BASE_DIR,
            data_type=config['data_type'],
            RM=RM
        )

        print(f"\n--- Part 2 completed for {RM} ---")
        print(f"Consensus motifs shape: {consensus_motif.shape}")

        # Store results
        result.update({
            'consensus_motif': consensus_motif,
            'ig_scores': ig_scores,
            'pwm_list': pwm_list,
            'status': 'success'
        })

        # Save motif logos immediately after processing completes
        if png_dir is not None:
            print(f"\n{'='*60}")
            print(f"Drawing motif logos for: {RM}")
            print(f"{'='*60}")
            try:
                # Save PNG format
                draw_motif_logos(
                    consensus_motif,
                    ig_scores,
                    RM,
                    res_dir=png_dir,
                    filename=RM,
                    save_only=True,
                    file_format='png'
                )

                # Save PDF format if pdf_dir is provided
                if pdf_dir is not None:
                    os.makedirs(pdf_dir, exist_ok=True)
                    draw_motif_logos(
                        consensus_motif,
                        ig_scores,
                        RM,
                        res_dir=pdf_dir,
                        filename=RM,
                        save_only=True,
                        file_format='pdf'
                    )

                print(f"Motif logo saved for {RM}")
            except Exception as e:
                print(f"Error saving motif logo for {RM}: {e}")
                traceback.print_exc()

        # Save Tomtom results immediately after processing completes
        if tomtom_dir is not None:
            print(f"\n{'='*60}")
            print(f"Saving Tomtom results for: {RM}")
            print(f"{'='*60}")
            try:
                tomtom_tsv_path = f"{BASE_DIR}tomtom_out/tomtom.tsv"

                # Create subdirectory for each modification
                rm_tomtom_dir = os.path.join(tomtom_dir, RM)
                os.makedirs(rm_tomtom_dir, exist_ok=True)

                # Save tomtom results to modification-specific subdirectory
                display_tomtom_results(
                    tomtom_tsv_path,
                    pwm_list,
                    res_dir=rm_tomtom_dir,
                    filename='results'  # Save as {RM}/results.txt
                )

                print(f"Tomtom results saved for {RM} to: {rm_tomtom_dir}/results.txt")
            except Exception as e:
                print(f"Error saving tomtom results for {RM}: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"\n!!! Error processing {RM}: {e} !!!")
        traceback.print_exc()
        result.update({
            'status': 'error',
            'error': str(e)
        })

    return result


def save_motif_logo(result, png_dir, pdf_dir=None):
    """
    Save motif logo for a modification in both PNG and PDF formats.

    Args:
        result: Result dictionary from process_single_modification
        png_dir: Directory to save PNG files
        pdf_dir: Directory to save PDF files (optional)

    Returns:
        Success status
    """
    RM = result['RM']

    print(f"\n{'='*60}")
    print(f"Drawing motif logos for: {RM}")
    print(f"{'='*60}")

    if result.get('status') != 'success':
        print(f"Skipping {RM} due to error in pipeline.")
        return False

    try:
        # Save PNG format
        draw_motif_logos(
            result['consensus_motif'],
            result['ig_scores'],
            RM,
            res_dir=png_dir,
            filename=RM,
            save_only=True,
            file_format='png'
        )

        # Save PDF format if pdf_dir is provided
        if pdf_dir is not None:
            os.makedirs(pdf_dir, exist_ok=True)
            draw_motif_logos(
                result['consensus_motif'],
                result['ig_scores'],
                RM,
                res_dir=pdf_dir,
                filename=RM,
                save_only=True,
                file_format='pdf'
            )

        print(f"Motif logo saved for {RM}")
        return True

    except Exception as e:
        print(f"Error saving motif logo for {RM}: {e}")
        traceback.print_exc()
        return False


def save_tomtom_results(result, tomtom_dir):
    """
    Save Tomtom results for a modification.

    Args:
        result: Result dictionary from process_single_modification
        tomtom_dir: Directory to save TXT files

    Returns:
        Success status
    """
    RM = result['RM']

    print(f"\n{'='*60}")
    print(f"Saving Tomtom results for: {RM}")
    print(f"{'='*60}")

    if result.get('status') != 'success':
        print(f"Skipping {RM} due to error in pipeline.")
        return False

    try:
        tomtom_tsv_path = f"{result['BASE_DIR']}tomtom_out/tomtom.tsv"

        # Create subdirectory for each modification
        rm_tomtom_dir = os.path.join(tomtom_dir, RM)
        os.makedirs(rm_tomtom_dir, exist_ok=True)

        # Save tomtom results to modification-specific subdirectory
        # Save as 'results.txt' inside the RM subdirectory
        display_tomtom_results(
            tomtom_tsv_path,
            result['pwm_list'],
            res_dir=rm_tomtom_dir,
            filename='results'  # Save as {RM}/results.txt
        )

        print(f"Tomtom results saved for {RM} to: {rm_tomtom_dir}/results.txt")
        return True

    except Exception as e:
        print(f"Error saving tomtom results for {RM}: {e}")
        traceback.print_exc()
        return False


def process_modifications_batch(modifications, config, eps_dict=None, png_dir=None, pdf_dir=None, tomtom_dir=None, dna=False):
    """
    Process a batch of modifications sequentially.

    This function is called in each child process, so it sets up logging.

    Args:
        modifications: List of modification names to process
        config: Configuration dictionary
        eps_dict: Dictionary mapping modification names to eps values
        png_dir: Directory to save PNG files
        pdf_dir: Directory to save PDF files
        tomtom_dir: Directory to save Tomtom result TXT files
        dna: If True, pass --dna flag to STREME

    Returns:
        List of result dictionaries
    """
    global logger, original_stdout, original_stderr

    # Set up logging in child process
    log_dir = 'draw_motif/draw_motif/logs'
    logger, original_stdout, original_stderr = setup_logging(log_dir, is_child_process=True)

    results = []

    for RM in modifications:
        result = process_single_modification(RM, config, eps_dict, png_dir, pdf_dir, tomtom_dir, dna)
        results.append(result)

    return results


def run_pipeline_parallel(modifications, config, num_processes=3, eps_dict=None, png_dir=None, pdf_dir=None, tomtom_dir=None, dna=False):
    """
    Run the pipeline for multiple modifications in parallel.

    Args:
        modifications: List of modification names to process
        config: Configuration dictionary
        num_processes: Number of parallel processes
        eps_dict: Dictionary mapping modification names to eps values
        png_dir: Directory to save PNG files
        pdf_dir: Directory to save PDF files
        tomtom_dir: Directory to save Tomtom result TXT files
        dna: If True, pass --dna flag to STREME

    Returns:
        List of result dictionaries from all processes
    """
    if num_processes <= 1:
        # Sequential processing
        return process_modifications_batch(modifications, config, eps_dict, png_dir, pdf_dir, tomtom_dir, dna)

    # Split modifications into batches for each process
    batch_size = (len(modifications) + num_processes - 1) // num_processes
    batches = [modifications[i:i+batch_size] for i in range(0, len(modifications), batch_size)]

    print(f"Processing {len(modifications)} modifications with {num_processes} processes")
    print(f"Batch sizes: {[len(b) for b in batches]}")

    # Create a pool of workers
    with mp.Pool(processes=num_processes) as pool:
        # Use partial to pass config, eps_dict, and output directories
        func = partial(process_modifications_batch, config=config, eps_dict=eps_dict, png_dir=png_dir, pdf_dir=pdf_dir, tomtom_dir=tomtom_dir, dna=dna)

        # Map batches to workers
        results_batches = pool.map(func, batches)

    # Flatten results
    results = []
    for batch_results in results_batches:
        results.extend(batch_results)

    return results


def auto_generate_eps(modifications):
    """
    Auto-generate eps values based on modification type.

    Uses DEFAULT_EPS_DICT for recommended eps values.

    Args:
        modifications: List of modification names

    Returns:
        Dictionary mapping modification names to eps values
    """
    eps_dict = {}
    for rm in modifications:
        if rm in DEFAULT_EPS_DICT:
            eps_dict[rm] = DEFAULT_EPS_DICT[rm]
        else:
            eps_dict[rm] = 0.3  # Default fallback
    return eps_dict


def parse_eps_list(eps_list_str, modifications=None, auto_eps=False):
    """
    Parse a list of eps values from command line argument.

    Args:
        eps_list_str: Comma-separated list of eps values (e.g., "0.3,0.5,0.3")
        modifications: List of modification names (required when auto_eps=True)
        auto_eps: If True and eps_list_str is None, auto-generate eps values

    Returns:
        Dictionary mapping modification names to eps values
    """
    # Auto-generate eps if requested
    if eps_list_str is None and auto_eps:
        if modifications is None:
            raise ValueError("modifications must be provided when auto_eps=True")
        return auto_generate_eps(modifications)

    # Use DEFAULT_EPS_DICT if no eps_list provided
    if eps_list_str is None:
        return DEFAULT_EPS_DICT.copy()

    eps_values = [float(x.strip()) for x in eps_list_str.split(',')]

    # If modifications are provided, apply eps values in order
    if modifications is not None:
        eps_dict = {}
        for i, rm in enumerate(modifications):
            if i < len(eps_values):
                eps_dict[rm] = eps_values[i]
            else:
                # Use default value from DEFAULT_EPS_DICT
                eps_dict[rm] = DEFAULT_EPS_DICT.get(rm, 0.3)
        return eps_dict

    # Otherwise, apply to DEFAULT_EPS_DICT keys in order (backward compatibility)
    eps_dict = {}
    default_mods = list(DEFAULT_EPS_DICT.keys())
    for i, rm in enumerate(default_mods):
        if i < len(eps_values):
            eps_dict[rm] = eps_values[i]
        else:
            eps_dict[rm] = DEFAULT_EPS_DICT[rm]

    return eps_dict


def main():
    parser = argparse.ArgumentParser(
        description='Multi-process motif drawing for all RNA modifications',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data parameters
    parser.add_argument(
        '--data-path',
        type=str,
        default='/home/dc/vscode/vscode20260112/MultiRM/data/MultiRM_data.h5',
        help='Path to HDF5 data file'
    )
    parser.add_argument(
        '--model-weights',
        type=str,
        default='/home/dc/vscode/vscode20260112/MultiRM/logs51/20260215_232022/epoch12.pkl',
        help='Path to model weights file'
    )
    parser.add_argument(
        '--data-type',
        type=str,
        default='train',
        choices=['train', 'valid', 'test'],
        help='Data type to use'
    )
    parser.add_argument(
        '--modifications',
        type=str,
        default=None,
        help='Comma-separated list of modifications (e.g., "hAm,hCm,hm6A"). Default: use DEFAULT_EPS_DICT keys. With --auto-eps, automatically generates eps values for specified modifications.'
    )

    # Pipeline parameters
    parser.add_argument(
        '-w',
        type=int,
        default=8,
        help='Window size for IG score aggregation'
    )
    parser.add_argument(
        '-k',
        type=int,
        default=5,
        help='Top-K windows to select per sample'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5000,
        help='Number of samples to process'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=666,
        help='Random seed for sample selection (default: 666)'
    )
    parser.add_argument(
        '--alignment-script',
        type=str,
        default='draw_motif/alignment_local.R',
        help='Path to R alignment script'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for model inference (default: 128)'
    )

    # Eps parameters
    parser.add_argument(
        '--eps-list',
        type=str,
        default=None,
        help='Comma-separated list of eps values (applied to modifications in order)'
    )
    parser.add_argument(
        '--auto-eps',
        action='store_true',
        help='Auto-generate eps values based on modification type (uses recommended values)'
    )
    parser.add_argument(
        '--dna',
        action='store_true',
        help='Use --dna flag for STREME (treats sequences as DNA with ACGT alphabet instead of RNA)'
    )

    # Multiprocessing parameters
    parser.add_argument(
        '--num-processes',
        type=int,
        default=3,
        help='Number of parallel processes to use'
    )

    # Output directories
    parser.add_argument(
        '--png-dir',
        type=str,
        default='draw_motif/draw_motif/png',
        help='Directory to save motif logo PNG files'
    )
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default='draw_motif/draw_motif/pdf',
        help='Directory to save motif logo PDF files'
    )
    parser.add_argument(
        '--tomtom-dir',
        type=str,
        default='draw_motif/draw_motif/tomtom',
        help='Directory to save Tomtom result TXT files'
    )

    args = parser.parse_args()

    # Parse modifications to process
    # Default: use keys from DEFAULT_EPS_DICT (allows control via DEFAULT_EPS_DICT)
    if args.modifications is None:
        modifications = list(DEFAULT_EPS_DICT.keys())
    else:
        modifications = [m.strip() for m in args.modifications.split(',')]
        # Validate modification names
        for m in modifications:
            if m not in ALL_MODIFICATIONS:
                print(f"Warning: Unknown modification '{m}'. Skipping.")
        modifications = [m for m in modifications if m in ALL_MODIFICATIONS]

    # Parse eps values and filter to only include selected modifications
    eps_dict = parse_eps_list(args.eps_list, modifications=modifications, auto_eps=args.auto_eps)
    # Filter eps_dict to only include selected modifications
    eps_dict = {rm: eps_dict[rm] for rm in modifications if rm in eps_dict}

    # Set up logging
    log_dir = 'draw_motif/draw_motif/logs'
    global logger, original_stdout, original_stderr
    logger, original_stdout, original_stderr = setup_logging(log_dir)

    # Create output directories
    os.makedirs(args.png_dir, exist_ok=True)
    os.makedirs(args.pdf_dir, exist_ok=True)
    os.makedirs(args.tomtom_dir, exist_ok=True)

    # Build configuration dictionary
    config = {
        'data_path': args.data_path,
        'model_weights': args.model_weights,
        'data_type': args.data_type,
        'w': args.w,
        'k': args.k,
        'sample_size': args.sample_size,
        'seed': args.seed,
        'alignment_script': args.alignment_script,
        'batch_size': args.batch_size
    }

    # Log pipeline configuration
    logger.info("=" * 80)
    logger.info("Motif Drawing Pipeline - All Modifications")
    logger.info("=" * 80)
    logger.info(f"Modifications to process: {modifications}")
    logger.info(f"Number of processes: {args.num_processes}")
    logger.info(f"Data type: {args.data_type}")
    logger.info(f"Window size (w): {args.w}, Top-K (k): {args.k}")
    logger.info(f"Sample size: {args.sample_size}, Seed: {args.seed}")
    logger.info(f"Inference batch size: {args.batch_size}")
    logger.info(f"PNG output: {args.png_dir}")
    logger.info(f"PDF output: {args.pdf_dir}")
    logger.info(f"Tomtom output: {args.tomtom_dir}")
    logger.info(f"Log directory: {log_dir}")
    eps_mode = "Auto-generated" if args.auto_eps else ("Manual (--eps-list)" if args.eps_list else "DEFAULT_EPS_DICT")
    logger.info(f"Eps mode: {eps_mode}")
    logger.info("Eps values:")
    for rm in modifications:
        logger.info(f"  {rm}: eps={eps_dict.get(rm, 0.3)}")
    logger.info(f"STREME mode: {'DNA (ACGT)' if args.dna else 'RNA (ACGU)'}")
    logger.info("=" * 80)

    # Run pipeline (motif logos and tomtom results are saved immediately after each modification completes)
    logger.info("Starting pipeline execution...")
    results = run_pipeline_parallel(
        modifications=modifications,
        config=config,
        num_processes=args.num_processes,
        eps_dict=eps_dict,
        png_dir=args.png_dir,
        pdf_dir=args.pdf_dir,
        tomtom_dir=args.tomtom_dir,
        dna=args.dna
    )

    logger.info("Pipeline execution completed")

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    success_count = sum(1 for r in results if r.get('status') == 'success')
    error_count = len(results) - success_count

    print(f"Total modifications processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")

    # Log summary
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Total modifications processed: {len(results)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Errors: {error_count}")

    if error_count > 0:
        print("\nErrors:")
        logger.info("Errors:")
        for r in results:
            if r.get('status') == 'error':
                error_msg = f"  {r['RM']}: {r.get('error', 'Unknown error')}"
                print(error_msg)
                logger.info(error_msg)

    logger.info(f"Motif logos saved to: {args.png_dir} (PNG)")
    logger.info(f"Motif logos saved to: {args.pdf_dir} (PDF)")
    logger.info(f"Tomtom results saved to: {args.tomtom_dir}")
    logger.info("=" * 80)

    print(f"\nMotif logos saved to: {args.png_dir} (PNG)")
    print(f"Motif logos saved to: {args.pdf_dir} (PDF)")
    print(f"Tomtom results saved to: {args.tomtom_dir}")
    print(f"Logs saved to: {log_dir}")
    print("=" * 80)
    # Restore original stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr



if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    main()
