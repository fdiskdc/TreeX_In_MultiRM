#!/usr/bin/env python3
"""
Utility functions for motif extraction and visualization pipeline.
"""

import numpy as np
import pandas as pd
import h5py
import os
import subprocess
import sys
import torch

from sklearn.cluster import DBSCAN
import umap
import logomaker
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import to_rgba

# Import functions from util_cm_treex_local
from util_cm_treex_local import (
    highest_score, highest_x, reduction_clustering, result_idx,
    pfm, pwm, read_seq, read_seq_all, to_onehot, helper,
    cal_consensus_motif_2
)

# Import data loading utilities from Scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scripts'))
from train_utils import RMdata


def load_model_and_data(data_path, model_weights, data_type, RM_index, length, sample_size, seed=666):
    """
    Load TreeX model and dataset.

    Note: sample_size is applied AFTER filtering positive samples in run_pipeline_part1.

    Args:
        data_path: Path to HDF5 data file
        model_weights: Path to model weights file
        data_type: 'train', 'valid', or 'test'
        RM_index: Index of the RNA modification type
        length: Sequence length
        sample_size: Number of positive samples to use (applied after filtering)
        seed: Random seed for reproducibility (default: 666)

    Returns:
        model, dataset, all_indices, device
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Import model here to avoid early import issues
    from main_model import RNA_ClassQuery_Model_Treex
    from train_utils import load_RM_data

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RNA_ClassQuery_Model_Treex(
        cnn_hidden_dim=128,
        cnn_kernel_sizes=(1, 3, 5, 7),
        cnn_dropout=0.1,
        num_classes=12,
        lstm_hidden_dim=256,
        lstm_num_layers=3,
        lstm_dropout=0.1,
        num_attn_heads=4,
        attn_dropout=0.1,
        use_hierarchical=True,
        use_simple_pooling=False,
        use_layer_norm=True,
        seq_len=length
    )

    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device)
    model.eval()

    # Load dataset
    dataset = RMdata(data_path, use_embedding=False, length=length, mode=data_type)

    # Return all indices - positive sample filtering happens later
    num_samples = len(dataset)
    all_indices = np.arange(num_samples)

    return model, dataset, all_indices, device


def get_predictions(model, dataset, selected_indices, RM_index, device, batch_size=128):
    """
    Get model predictions for selected samples.

    Args:
        model: Trained TreeX model
        dataset: Dataset object
        selected_indices: Indices of samples to process
        RM_index: Index of RNA modification type
        device: torch device
        batch_size: Batch size for inference (default: 128)

    Returns:
        predictions, labels, sequences
    """
    predictions = []
    labels = []
    sequences = []

    with torch.no_grad():
        # Process in batches
        for i in range(0, len(selected_indices), batch_size):
            batch_indices = selected_indices[i:i + batch_size]
            batch_x = []
            batch_y = []

            for idx in batch_indices:
                x, y = dataset[idx]
                batch_x.append(x)
                batch_y.append(y)

            # Stack batch and convert to tensor
            # Check if batch_x elements are 1D (flattened) or 2D (4, length)
            first_x = batch_x[0]
            if isinstance(first_x, torch.Tensor):
                if first_x.ndim == 1:
                    # 1D tensors: stack creates (batch_size, 4*length)
                    batch_x = torch.stack(batch_x).to(device)  # (batch_size, 4*length)
                    # Reshape to (batch_size, 4, length) for proper handling
                    seq_length = batch_x.shape[1] // 4
                    batch_x = batch_x.view(batch_x.size(0), 4, seq_length)  # (batch_size, 4, length)
                else:
                    # Already 2D tensors
                    batch_x = torch.stack(batch_x).to(device)  # (batch_size, 4, length)
            else:
                # Convert to tensor then stack
                batch_x = torch.stack([torch.tensor(x) for x in batch_x]).to(device)
                if batch_x.ndim == 2:
                    seq_length = batch_x.shape[1] // 4
                    batch_x = batch_x.view(batch_x.size(0), 4, seq_length)

            batch_x_in = batch_x.transpose(1, 2)  # (batch_size, length, 4) for model input

            # Forward pass
            pred = model(batch_x_in)
            if isinstance(pred, tuple):
                pred = pred[0]
            if isinstance(pred, list):
                pred = torch.stack(pred, dim=1)

            # Extract predictions for RM_index
            batch_preds = pred[:, RM_index].cpu().numpy()  # (batch_size,)

            # Extract labels and sequences - keep (4, length) shape
            for j, idx in enumerate(batch_indices):
                predictions.append(batch_preds[j])
                labels.append(batch_y[j].numpy()[RM_index])
                # batch_x[j] is now (4, length) shape
                sequences.append(batch_x[j].cpu().numpy())

    predictions = np.array(predictions)
    labels = np.array(labels)
    # sequences should now be a list of (4, length) arrays, create 3D array
    sequences = np.array(sequences)

    return predictions, labels, sequences


def get_positive_label_samples(predictions, labels, sample_size=None, seed=666):
    """
    Get indices of samples with positive true labels, optionally sampling and sorting by prediction confidence.

    This follows the consensus_motifs_cp1_treex.ipynb approach:
    1. Filter samples where true label == 1 (actual positive samples)
    2. Optionally sample from these to control sample size
    3. Sort by prediction score (descending)
    4. Take all sampled samples (they are already sorted by confidence)

    Args:
        predictions: Model predictions (confidence scores)
        labels: True labels (0 or 1)
        sample_size: Number of samples to use (None = use all positive samples)
        seed: Random seed for sampling

    Returns:
        Indices of selected positive samples
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Filter samples where true label is 1 (these are true positive samples by label)
    positive_indices = np.where(labels == 1)[0]

    if len(positive_indices) == 0:
        print(f"Warning: No samples with positive labels found")
        return positive_indices

    print(f"Total positive label samples: {len(positive_indices)}")

    # Optionally sample from positive samples
    if sample_size is not None and sample_size < len(positive_indices):
        positive_indices = np.random.choice(positive_indices, sample_size, replace=False)
        print(f"Randomly sampled {sample_size} / {len(positive_indices) if len(positive_indices) > sample_size else len(positive_indices)} positive samples")
    else:
        print(f"Using all {len(positive_indices)} positive samples")

    # Sort by prediction score (descending) - most confident first
    pred_scores = predictions[positive_indices]
    sorted_order = np.argsort(-pred_scores)  # Negative for descending sort
    positive_indices = positive_indices[sorted_order]

    return positive_indices


def get_true_positive_samples(predictions, labels, threshold=0.5):
    """
    Get indices of true positive samples.

    Args:
        predictions: Model predictions
        labels: True labels
        threshold: Prediction threshold

    Returns:
        Indices of true positive samples
    """
    tp_mask = (predictions >= threshold) & (labels == 1)
    tp_indices = np.where(tp_mask)[0]
    return tp_indices


def fast_batched_ig_treex(model, x, y_true, RM_index, device, num_steps=50):
    """
    Compute Integrated Gradients for a sample.

    Args:
        model: TreeX model
        x: Input sequence [1, 4, length]
        y_true: True label
        RM_index: Task index
        device: torch device
        num_steps: Number of steps for IG approximation

    Returns:
        IG scores as 1D numpy array
    """
    from captum.attr import IntegratedGradients

    # Debug: check input shape
    # print(f"IG input shape: {x.shape}")

    # Handle model output format
    def forward_func(x_input):
        output = model(x_input)
        # Debug: check model output
        # print(f"Model output type: {type(output)}, shape: {output[0].shape if isinstance(output, (list, tuple)) else output.shape}")
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, list):
            output = torch.stack(output, dim=1)
        # Return 2D tensor [batch, 1] for Captum
        result = output[:, RM_index:RM_index+1]
        # print(f"Forward func output shape: {result.shape}")
        return result

    ig = IntegratedGradients(forward_func)

    x_input = x.requires_grad_(True)

    with torch.no_grad():
        # Create baseline with all zeros (representing no signal)
        baseline = torch.zeros_like(x_input)

    # Save original training mode and set to training mode
    # LSTM layers require training mode for backward pass
    was_training = model.training
    model.train()

    # Disable dropout during IG computation for stability
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    try:
        # For multi-label classification, don't use target parameter
        # The forward_func already selects the specific task output
        with torch.enable_grad():
            attributions = ig.attribute(x_input, baselines=baseline, n_steps=num_steps)

        # Sum across channels to get per-position importance
        # attributions shape: [batch_size, 4, seq_length] -> sum to [batch_size, seq_length]
        ig_scores = attributions.sum(dim=1)

        # Detach from computation graph before converting to numpy
        ig_scores = ig_scores.detach()

        # Ensure we get a 1D array
        if ig_scores.dim() > 1:
            ig_scores = ig_scores.squeeze()
        if ig_scores.dim() == 0:
            # Single value, convert to 1D array
            ig_scores = ig_scores.unsqueeze(0)

        ig_scores = ig_scores.cpu().numpy()

        # Ensure it's a 1D array
        ig_scores = np.atleast_1d(ig_scores).flatten()

        # Debug: check IG scores
        # print(f"IG scores computed: shape={ig_scores.shape}, sum={np.sum(np.abs(ig_scores)):.6f}")

    except Exception as e:
        import traceback
        print(f"Warning: IG computation error: {e}")
        traceback.print_exc()
        # Return zeros with expected sequence length
        seq_length = x.shape[-1]
        ig_scores = np.zeros(seq_length)

    finally:
        # Restore original training mode
        model.train(was_training)

        # Restore dropout probabilities
        dropout_p = 0.1  # Default dropout from model config
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_p

    return ig_scores


def compute_ig(model, sequences, labels, RM_index, device, sample_indices=None):
    """
    Compute IG scores for selected samples.

    Args:
        model: TreeX model
        sequences: Input sequences - list of arrays with shape (4, length)
        labels: True labels
        RM_index: Task index
        device: torch device
        sample_indices: Indices to compute IG for

    Returns:
        IG scores array (n_samples, seq_length) or empty array if no samples
    """
    ig_scores = []

    if sample_indices is None:
        sample_indices = range(len(sequences))

    # Convert to list to avoid numpy indexing issues
    if not isinstance(sequences, list):
        sequences = list(sequences)
    if not isinstance(labels, list):
        labels = list(labels)

    for idx in sample_indices:
        # sequences are stored as (4, length) from get_predictions
        seq = sequences[idx]  # Shape: (4, length)
        seq = np.asarray(seq)  # Ensure it's a numpy array

        # Handle different input shapes
        if seq.ndim == 1:
            # Unexpected 1D array, skip or handle
            print(f"Warning: Sequence {idx} is 1D with shape {seq.shape}, skipping")
            continue
        elif seq.ndim == 2:
            seq_len = seq.shape[1]
        else:
            print(f"Warning: Sequence {idx} has unexpected shape {seq.shape}, skipping")
            continue

        # Convert to tensor and ensure correct shape (1, 4, length)
        x = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, 4, length)
        y = int(labels[idx])

        try:
            ig = fast_batched_ig_treex(model, x, y, RM_index, device)
            # Ensure ig is a 1D numpy array
            ig = np.asarray(ig).flatten()
            # Ensure it matches sequence length
            if len(ig) != seq_len:
                if len(ig) < seq_len:
                    # Pad with zeros
                    ig = np.pad(ig, (0, seq_len - len(ig)), mode='constant')
                else:
                    # Truncate
                    ig = ig[:seq_len]
            ig_scores.append(ig)
        except Exception as e:
            print(f"Warning: IG computation failed for sample {idx}: {e}")
            ig_scores.append(np.zeros(seq_len))

    if len(ig_scores) == 0:
        return np.array([])

    return np.array(ig_scores)


def extract_sequences(sequences, ig_scores, nucleos_df, w=8, k=5, p=1):
    """
    Extract short sequences based on IG scores.

    Args:
        sequences: Input sequences (n_samples, seq_length, 4) one-hot encoded
        ig_scores: IG scores (n_samples, seq_length)
        nucleos_df: DataFrame with nucleotide sequences
        w: Window size
        k: Top-K windows
        p: Padding

    Returns:
        short_seqs, scores, raw_seqs
    """
    # Check if we have data to process
    if len(sequences) == 0:
        print("Warning: No sequences to extract")
        return [], [], []

    # Convert ig_scores to numpy array if it's a tuple or list
    ig_scores = np.asarray(ig_scores)

    if len(ig_scores) == 0:
        print("Warning: No IG scores to extract")
        return [], [], []

    # Ensure ig_scores is 2D
    ig_scores = np.atleast_2d(ig_scores)

    # Check if IG scores are all zeros or very small
    ig_sum = np.sum(np.abs(ig_scores))
    if ig_sum < 1e-10:
        print("Warning: IG scores are all zeros or extremely small!")
        print("This indicates that:")
        print("  1. IG computation may have failed")
        print("  2. Model output may not have gradients")
        print("  3. Model weights may be incompatible")
        print(f"IG scores shape: {ig_scores.shape}, sum: {ig_sum}")
        print("First few IG scores:", ig_scores[0, :5] if ig_scores.size > 5 else ig_scores.flatten()[:5])

    # Convert one-hot to nucleotides
    # sequences have shape (4, length) after transpose in get_predictions
    nucleotides = ['A', 'C', 'G', 'T']

    raw_seqs = []
    for seq in sequences:
        seq_str = ''
        # seq shape is (4, length), so iterate over length
        for pos in range(seq.shape[1]):
            nuc_idx = np.argmax(seq[:, pos])  # Get argmax over 4 channels
            seq_str += nucleotides[nuc_idx]
        raw_seqs.append(seq_str)

    # Use helper function to extract short sequences
    # RM_array shape: (n_samples, seq_length, 1)
    try:
        RM_array = ig_scores.reshape(-1, ig_scores.shape[1], 1)
    except (IndexError, ValueError) as e:
        print(f"Warning: Cannot reshape ig_scores with shape {ig_scores.shape}: {e}")
        # Handle case where ig_scores is 1D or has unexpected shape
        if ig_scores.ndim == 1:
            RM_array = ig_scores.reshape(-1, len(ig_scores), 1)
        else:
            print("Warning: Unable to extract sequences due to invalid IG scores shape")
            return [], [], raw_seqs

    short_seqs, scores = helper(RM_array, nucleos_df, 'motif', 51, 'motif',
                                download=False, w=w, k=k, p=p)

    return short_seqs, scores, raw_seqs


def save_to_fasta(sequences, output_path, names=None):
    """
    Save sequences to FASTA format.

    Args:
        sequences: List of sequences
        output_path: Output file path
        names: Optional list of sequence names
    """
    with open(output_path, 'w') as f:
        for i, seq in enumerate(sequences):
            if names is not None and i < len(names):
                f.write(f">{names[i]}\n")
            else:
                f.write(f">seq{i}\n")
            f.write(f"{seq}\n")

    print(f"Saved {len(sequences)} sequences to {output_path}")


def run_alignment(sequences, output_dir, script_path, max_length=200):
    """
    Run sequence alignment using R script.

    Args:
        sequences: Sequences to align
        output_dir: Output directory
        script_path: Path to alignment script
        max_length: Maximum sequence length for alignment

    Returns:
        aligned_seqs
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter sequences by length
    filtered_seqs = [s for s in sequences if len(s) <= max_length]

    if len(filtered_seqs) < len(sequences):
        print(f"Filtered {len(sequences) - len(filtered_seqs)} sequences longer than {max_length}")

    # Create input FASTA
    input_fasta = os.path.join(output_dir, 'input.fasta')
    save_to_fasta(filtered_seqs, input_fasta)

    # Run alignment script
    output_fasta = os.path.join(output_dir, 'aligned.fasta')

    cmd = ['Rscript', script_path, input_fasta, output_fasta]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Alignment completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Alignment error: {e.stderr}")
        # Return original sequences if alignment fails
        return sequences

    # Read aligned sequences
    aligned_seqs = []
    with open(output_fasta, 'r') as f:
        current_seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    aligned_seqs.append(current_seq)
                current_seq = ''
            else:
                current_seq += line
        if current_seq:
            aligned_seqs.append(current_seq)

    return aligned_seqs


def run_streme(output_dir, positive_fasta, negative_fasta=None, minw=5, maxw=15, dna=False, conda_env='meme_env'):
    """
    Run STREME motif discovery.

    Args:
        output_dir: Output directory
        positive_fasta: Path to positive sequences FASTA
        negative_fasta: Optional path to negative sequences FASTA
        minw: Minimum motif width
        maxw: Maximum motif width
        dna: If True, use --dna flag (treats sequences as DNA with ACGT alphabet)
        conda_env: Conda environment name where STREME is installed (default: 'meme_env')

    Returns:
        Path to STREME output
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use conda run to execute streme in the specified environment
    # Note: STREME requires --p for positive file and --n for negative file
    streme_cmd = [
        'conda', 'run', '-n', conda_env,
        'streme',
        '--p', positive_fasta,
        '--minw', str(minw),
        '--maxw', str(maxw),
        '--oc', output_dir
    ]

    if negative_fasta:
        streme_cmd.extend(['--n', negative_fasta])

    if dna:
        streme_cmd.append('--dna')

    try:
        result = subprocess.run(streme_cmd, capture_output=True, text=True, check=True)
        print("STREME completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"STREME error: {e.stderr}")
        raise

    return output_dir


def auto_search_best_eps(aligned_seqs, scores, eps_range=[0.1, 3.0], eps_step=0.1,
                        min_clusters=2, max_clusters=20):
    """
    Auto-search for best DBSCAN eps parameter.

    Args:
        aligned_seqs: Aligned sequences
        scores: IG scores
        eps_range: Range of eps values to search
        eps_step: Step size for eps search
        min_clusters: Minimum number of clusters
        max_clusters: Maximum number of clusters

    Returns:
        best_eps, cluster_analysis_df
    """
    eps_values = np.arange(eps_range[0], eps_range[1] + eps_step, eps_step)
    results = []

    print(f"Searching for best eps in range [{eps_range[0]}, {eps_range[1]}] with step {eps_step}")

    for eps in eps_values:
        try:
            consensus_motif, ig_score_list = cal_consensus_motif_2(
                aligned_seqs, scores, eps=eps
            )

            # Count clusters (non-noise clusters)
            n_clusters = len(ig_score_list)

            # Estimate noise points
            data = []
            for i in range(len(aligned_seqs)):
                tmp = to_onehot(aligned_seqs[i]).T.flatten()
                data.append(tmp)

            df = pd.DataFrame(data=data, index=list(range(len(aligned_seqs))))
            class_labels = reduction_clustering(df, n_clusters=6, eps=eps)
            n_noise = np.sum(class_labels == -1)

            results.append({
                'eps': eps,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            })

            print(f"eps={eps:.1f}: {n_clusters} clusters, {n_noise} noise points")

        except Exception as e:
            print(f"eps={eps:.1f}: Error - {e}")
            results.append({
                'eps': eps,
                'n_clusters': 0,
                'n_noise': len(aligned_seqs)
            })

    cluster_analysis_df = pd.DataFrame(results)

    # Select best eps (prefer reasonable number of clusters, low noise)
    valid_results = cluster_analysis_df[
        (cluster_analysis_df['n_clusters'] >= min_clusters) &
        (cluster_analysis_df['n_clusters'] <= max_clusters)
    ]

    if len(valid_results) > 0:
        # Choose eps with minimum noise among valid results
        best_idx = valid_results['n_noise'].idxmin()
        best_eps = cluster_analysis_df.loc[best_idx, 'eps']
    else:
        # Default to middle of range
        best_eps = (eps_range[0] + eps_range[1]) / 2

    print(f"\nBest eps selected: {best_eps}")

    return best_eps, cluster_analysis_df


def compute_consensus_motifs(aligned_seqs, scores, eps, BASE_DIR, data_type, RM):
    """
    Compute consensus motifs.

    Args:
        aligned_seqs: Aligned sequences
        scores: IG scores
        eps: DBSCAN eps parameter
        BASE_DIR: Base directory for outputs
        data_type: Data type
        RM: Modification type

    Returns:
        consensus_motif, ig_scores, pwm_list, used_eps
    """
    consensus_motif, ig_scores = cal_consensus_motif_2(aligned_seqs, scores, eps=eps)

    # Convert to list of PWMs (each motif is L x 4)
    pwm_list = [consensus_motif[i].T for i in range(consensus_motif.shape[0])]

    # Save consensus motifs
    npz_path = os.path.join(BASE_DIR, 'treex_consensus_motifs.npz')
    np.savez(npz_path, pwms=pwm_list)
    print(f"Consensus motifs saved to {npz_path}")

    return consensus_motif, ig_scores, pwm_list, eps


def save_consensus_motifs(consensus_motif, ig_scores, output_path):
    """
    Save consensus motifs to NPZ file.

    Args:
        consensus_motif: Consensus motif array
        ig_scores: IG scores
        output_path: Output file path
    """
    pwm_list = [consensus_motif[i].T for i in range(consensus_motif.shape[0])]
    np.savez(output_path, pwms=pwm_list, ig_scores=ig_scores)
    print(f"Consensus motifs saved to {output_path}")


def trim(motif, threshold=0.2):
    """
    Trim edge nucleotides with low information content.

    Args:
        motif: PWM matrix (L x 4)
        threshold: Threshold for trimming

    Returns:
        Trimmed motif
    """
    motif = np.array(motif)

    # Calculate information content at each position
    ic = np.log2((motif + 1e-10) / 0.25) * motif
    total_ic = np.sum(ic, axis=1)

    # Find first and last positions above threshold
    above_threshold = np.where(total_ic > threshold)[0]

    if len(above_threshold) == 0:
        return motif

    start = above_threshold[0]
    end = above_threshold[-1] + 1

    return motif[start:end]


def draw_motif_logos(consensus_motif, ig_scores, RM, res_dir=None, filename=None, save_only=False, file_format='png'):
    """
    Draw motif logos using logomaker with Liquid Glass visual style.

    Args:
        consensus_motif: Consensus motif array (n_motifs, 4, length)
        ig_scores: IG scores for each motif
        RM: Modification type name
        res_dir: Directory to save images (optional)
        filename: Output filename (optional)
        save_only: If True, only save without displaying
        file_format: Output file format, 'png' or 'pdf' (default: 'png')
    """
    n_motifs = consensus_motif.shape[0]

    if save_only and res_dir is None:
        save_only = False

    if save_only:
        os.makedirs(res_dir, exist_ok=True)
        output_path = os.path.join(res_dir, f"{filename}.{file_format}")

    # Calculate number of rows and columns for subplot
    n_cols = min(4, n_motifs)
    n_rows = (n_motifs + n_cols - 1) // n_cols

    # Liquid Glass Color Scheme - Morandi colors with transparency
    color_scheme = {
        'A': to_rgba('#9BA4B5', 0.80),  # Muted blue-gray
        'C': to_rgba('#C8B8C0', 0.80),  # Muted mauve/pink-gray
        'G': to_rgba('#A8B5A2', 0.80),  # Muted sage green-gray
        'U': to_rgba('#D4C4B0', 0.80),  # Muted sand/beige-gray (RNA uses U instead of T)
    }

    # Create figure with frosted glass background
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 2.5 * n_rows),
                            facecolor='#F5F7FA')  # Light frosted gray background

    if n_motifs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_motifs):
        ax = axes[i]

        # Get PWM and transpose to (length, 4) for logomaker
        pwm = consensus_motif[i].T

        # Trim low-information edges
        pwm_trimmed = trim(pwm, threshold=0.1)

        # Create DataFrame for logomaker
        df = pd.DataFrame(pwm_trimmed, columns=['A', 'C', 'G', 'U'])  # RNA uses U instead of T

        # Apply frosted glass background to each subplot
        ax.set_facecolor('#EBEDF0')

        # Remove all spines for minimalist look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Draw logo with custom color scheme
        try:
            logo = logomaker.Logo(df, ax=ax, color_scheme=color_scheme)

            # Apply Liquid Glass 3D effects to all glyphs
            for glyph in logo.ax.get_children():
                # Apply effects to patches (polygons representing letters)
                if hasattr(glyph, 'get_path') or isinstance(glyph, plt.Polygon):
                    # Create layered path effects for 3D glass appearance

                    # 1. Drop Shadow (downward offset, semi-transparent)
                    shadow = path_effects.SimplePatchShadow(
                        offset=(2, -2),
                        alpha=0.3
                    )

                    # 2. Highlight Edge (bright stroke for glass refraction)
                    highlight = path_effects.Stroke(
                        linewidth=1.5,
                        foreground='white',
                        alpha=0.5
                    )

                    # 3. Subtle outer glow for depth
                    glow = path_effects.Stroke(
                        linewidth=2.5,
                        foreground='#8B9DAD',
                        alpha=0.2
                    )

                    # Combine effects: glow -> shadow -> normal -> highlight
                    glass_effects = [glow, shadow, path_effects.Normal(), highlight]

                    try:
                        glyph.set_path_effects(glass_effects)
                    except (AttributeError, TypeError):
                        # Some elements may not support path_effects
                        pass

                # Apply subtle glow effect to text elements
                elif isinstance(glyph, plt.Text):
                    text_glow = path_effects.withStroke(
                        linewidth=1.5,
                        foreground='white',
                        alpha=0.3
                    )
                    try:
                        glyph.set_path_effects([text_glow])
                    except (AttributeError, TypeError):
                        pass

        except Exception as e:
            print(f"Warning: Could not draw motif {i}: {e}")
            ax.set_facecolor('#EBEDF0')
            ax.text(0.5, 0.5, f"Motif {i+1}\n(Error)", ha='center', va='center',
                   fontsize=12, color='#5A6B7C', fontfamily='sans-serif')

        # Modern title with glass aesthetic
        ax.set_title(f"{RM} Motif {i+1}",
                    fontsize=13,
                    fontweight='bold',
                    fontfamily='sans-serif',
                    color='#3D4752',
                    pad=12,
                    loc='left')

        # Add subtle border for glass panel effect
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Add a very subtle bottom border
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#D1D9E6')
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['bottom'].set_alpha(0.5)

    # Hide extra subplots with consistent background
    for i in range(n_motifs, len(axes)):
        axes[i].axis('off')
        axes[i].set_facecolor('#F5F7FA')

    plt.tight_layout(pad=1.5, w_pad=1.2, h_pad=1.8)

    if save_only:
        # Set save parameters based on file format
        save_kwargs = {
            'bbox_inches': 'tight',
            'facecolor': '#F5F7FA',
            'edgecolor': 'none'
        }
        # Only add dpi for raster formats (png, jpg, etc.)
        if file_format in ['png', 'jpg', 'jpeg', 'tiff', 'svg']:
            save_kwargs['dpi'] = 200

        plt.savefig(output_path, **save_kwargs)
        plt.close()
        print(f"Motif logo saved to {output_path}")
    else:
        plt.show()


def run_tomtom_validation(query_meme, target_meme, output_dir, dist_method='pearson', conda_env='meme_env'):
    """
    Run Tomtom validation.

    Args:
        query_meme: Path to query MEME file
        target_meme: Path to target MEME file
        output_dir: Output directory
        dist_method: Distance method
        conda_env: Conda environment name where Tomtom is installed (default: 'meme_env')

    Returns:
        Path to Tomtom TSV file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use conda run to execute tomtom in the specified environment
    tomtom_cmd = [
        'conda', 'run', '-n', conda_env, '--no-capture-output',
        'tomtom',
        '-no-ssc',
        '-oc', output_dir,
        '-dist', dist_method,
        query_meme,
        target_meme
    ]

    try:
        result = subprocess.run(tomtom_cmd, capture_output=True, text=True, check=True)
        print("Tomtom completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Tomtom error: {e.stderr}")
        raise

    return os.path.join(output_dir, 'tomtom.tsv')


def export_to_meme(pwm_list, output_path, motif_names=None, bg_frequencies=None):
    """
    Export PWM list to MEME format.

    Args:
        pwm_list: List of PWM matrices (L x 4)
        output_path: Output file path
        motif_names: Optional list of motif names
        bg_frequencies: Background frequencies
    """
    if bg_frequencies is None:
        bg_frequencies = {'A': 0.295, 'C': 0.205, 'G': 0.205, 'T': 0.295}

    if motif_names is None:
        motif_names = [f"Model_Motif_{i+1}" for i in range(len(pwm_list))]

    pseudocount = 0.001

    with open(output_path, 'w') as f:
        f.write("MEME version 5.0\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("Background letter frequencies\n")
        f.write(f"A {bg_frequencies['A']} C {bg_frequencies['C']} G {bg_frequencies['G']} T {bg_frequencies['T']}\n\n")

        for idx, pwm in enumerate(pwm_list):
            pwm = np.array(pwm)

            if pwm.ndim != 2 or pwm.shape[1] != 4:
                raise ValueError(f"Motif {idx} has invalid shape {pwm.shape}")

            row_sums = np.sum(pwm, axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-3):
                total_counts = np.sum(pwm, axis=1, keepdims=True)
                pwm = (pwm + pseudocount) / (total_counts + 4 * pseudocount)

            motif_name = motif_names[idx] if idx < len(motif_names) else f"Model_Motif_{idx+1}"
            f.write(f"MOTIF {motif_name}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {pwm.shape[0]} nsites= 100\n")

            for row in pwm:
                f.write(" " + " ".join([f"{val:.6f}" for val in row]) + "\n")

            f.write("\n")

    print(f"MEME file written to: {output_path}")


def display_tomtom_results(tomtom_tsv_path, pwm_list, res_dir=None, filename=None, evalue_threshold=0.05):
    """
    Display and optionally save Tomtom validation results.

    Args:
        tomtom_tsv_path: Path to Tomtom TSV file
        pwm_list: List of PWM matrices
        res_dir: Directory to save results (optional)
        filename: Output filename (optional)
        evalue_threshold: E-value threshold for significance

    Returns:
        Significant matches
    """
    if not os.path.exists(tomtom_tsv_path):
        print(f"Tomtom results not found at: {tomtom_tsv_path}")
        return []

    significant_matches = []

    with open(tomtom_tsv_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        print("No Tomtom results found")
        return significant_matches

    header = lines[0].strip().split('\t')

    print("\n" + "=" * 60)
    print("Tomtom Validation Results")
    print("=" * 60)
    print(f"\nSignificant matches (E-value < {evalue_threshold}):")
    print("-" * 60)

    output_lines = []

    for line in lines[1:]:
        fields = line.strip().split('\t')

        if len(fields) < 4:
            continue

        try:
            query_id = fields[0]
            target_id = fields[1]
            evalue = float(fields[3])
            overlap = fields[5] if len(fields) > 5 else 'N/A'

            if evalue < evalue_threshold:
                match_info = {
                    'query': query_id,
                    'target': target_id,
                    'evalue': evalue,
                    'overlap': overlap
                }
                significant_matches.append(match_info)

                line_str = f"{query_id:20s} -> {target_id:20s}  E-value: {evalue:.6f}  Overlap: {overlap}"
                print(line_str)
                output_lines.append(line_str)

        except (ValueError, IndexError):
            continue

    print("-" * 60)
    print(f"Total motifs discovered by STREME: {len(pwm_list)}")
    print(f"Tree-X consensus motifs: {len([m for m in significant_matches])}")
    print(f"Significant matches: {len(significant_matches)}")
    print("=" * 60)

    # Save results if requested
    if res_dir is not None and filename is not None:
        os.makedirs(res_dir, exist_ok=True)
        output_path = os.path.join(res_dir, f"{filename}.txt")

        with open(output_path, 'w') as f:
            f.write("Tomtom Validation Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Significant matches (E-value < {evalue_threshold}):\n")
            f.write("-" * 60 + "\n")
            for line in output_lines:
                f.write(line + "\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total motifs discovered by STREME: {len(pwm_list)}\n")
            f.write(f"Tree-X consensus motifs: {len([m for m in significant_matches])}\n")
            f.write(f"Significant matches: {len(significant_matches)}\n")
            f.write("=" * 60 + "\n")

        print(f"Tomtom results saved to: {output_path}")

    return significant_matches


def run_pipeline_part1(data_path, model_weights, data_type, RM, RM_index, length,
                       w, k, auto_search_eps=False, eps_search_range=[0.1, 3],
                       eps_step=0.1, sample_size=5000, BASE_DIR=None,
                       alignment_script=None, eps=None, seed=666, batch_size=128, dna=False):
    """
    Run Part 1 of the pipeline: Data loading, IG computation, sequence extraction,
    alignment, STREME, and (optional) eps search.

    Args:
        data_path: Path to HDF5 data file
        model_weights: Path to model weights
        data_type: 'train', 'valid', or 'test'
        RM: RNA modification type name
        RM_index: Index of RNA modification
        length: Sequence length
        w: Window size for IG aggregation
        k: Top-K windows to select
        auto_search_eps: Whether to auto-search for best eps
        eps_search_range: Range for eps search
        eps_step: Step size for eps search
        sample_size: Number of samples to process (positive samples are sampled)
        BASE_DIR: Base directory for outputs
        alignment_script: Path to alignment script
        eps: Fixed eps value (skips auto-search if specified)
        seed: Random seed for sample selection (default: 666)
        batch_size: Batch size for model inference (default: 128)
        dna: If True, pass --dna flag to STREME (treats sequences as DNA with ACGT alphabet)

    Returns:
        aligned_short_seqs, scores, best_eps, cluster_analysis_df
    """
    if BASE_DIR is None:
        BASE_DIR = f"draw_motif/{data_type}_{RM}_w{w}_k{k}/"

    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(f"{BASE_DIR}streme_out", exist_ok=True)
    os.makedirs(f"{BASE_DIR}tomtom_out", exist_ok=True)

    print(f"{'='*60}")
    print(f"Running Part 1 for {RM}")
    print(f"{'='*60}")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"sample_size: {sample_size}, seed: {seed}")

    # Load model and data
    print("\n1. Loading model and data...")
    model, dataset, selected_indices, device = load_model_and_data(
        data_path, model_weights, data_type, RM_index, length, sample_size, seed
    )

    # Get predictions and positive label samples
    print(f"2. Getting predictions and selecting positive label samples (batch_size={batch_size})...")
    predictions, labels, sequences = get_predictions(model, dataset, selected_indices, RM_index, device, batch_size=batch_size)

    # Use the consensus_motifs_cp1_treex.ipynb approach:
    # 1. Filter samples where true label == 1
    # 2. Sample from these (using sample_size)
    # 3. Sort by prediction confidence (descending)
    tp_indices = get_positive_label_samples(predictions, labels, sample_size=sample_size, seed=seed)

    if len(tp_indices) == 0:
        print(f"Error: No positive label samples found for {RM}")
        print("Cannot proceed with motif extraction. Please check:")
        print("  - Model weights are correct")
        print("  - RM_index is correct")
        print("  - Data contains positive samples for this modification")
        raise ValueError(f"No positive label samples found for {RM}")

    if len(tp_indices) < 10:
        print(f"Warning: Very few positive samples ({len(tp_indices)})")

    # Use numpy array indexing directly to preserve shape
    tp_indices_list = tp_indices.tolist() if hasattr(tp_indices, 'tolist') else list(tp_indices)

    # Compute IG scores
    print("3. Computing Integrated Gradients...")
    print(f"  sequences shape: {sequences.shape}")
    print(f"  Number of positive samples: {len(tp_indices_list)}")
    # Use numpy indexing to get list of arrays with correct shape (4, length)
    selected_sequences = [sequences[i].copy() for i in tp_indices_list]
    selected_labels = [labels[i] for i in tp_indices_list]
    print(f"  selected_sequences[0] shape: {selected_sequences[0].shape if len(selected_sequences) > 0 else 'N/A'}")
    ig_scores = compute_ig(model, selected_sequences, selected_labels, RM_index, device)

    print(f"IG scores shape: {ig_scores.shape}")

    # Create nucleos DataFrame for helper function
    nucleos_data = []
    nucleotides = ['A', 'C', 'G', 'T']
    for seq in selected_sequences:
        seq_str = ''
        # seq shape is (4, length), so iterate over length
        for pos in range(seq.shape[1]):
            nuc_idx = np.argmax(seq[:, pos])  # Get argmax over 4 channels
            seq_str += nucleotides[nuc_idx]
        nucleos_data.append(list(seq_str))

    nucleos_df = pd.DataFrame(nucleos_data)

    # Extract short sequences
    print("4. Extracting short sequences based on IG scores...")
    short_seqs, seq_scores, raw_seqs = extract_sequences(
        selected_sequences, ig_scores, nucleos_df, w=w, k=k, p=1
    )

    print(f"Extracted {len(short_seqs)} short sequences")

    if len(short_seqs) == 0:
        print(f"Error: No short sequences extracted for {RM}")
        print("This could be due to:")
        print("  - IG scores are all zeros")
        print("  - Window size (w) or top-k (k) parameters are inappropriate")
        raise ValueError(f"No short sequences extracted for {RM}")

    # Save positive and negative sequences
    print("5. Saving sequences to FASTA...")
    save_to_fasta(short_seqs, os.path.join(BASE_DIR, 'positives.fasta'))

    # Run alignment
    print("6. Running sequence alignment...")
    if alignment_script is None:
        alignment_script = os.path.join(os.path.dirname(__file__), 'alignment_local.R')

    aligned_seqs = run_alignment(short_seqs, BASE_DIR, alignment_script)

    print(f"Aligned {len(aligned_seqs)} sequences")

    # Run STREME
    print("7. Running STREME motif discovery...")
    run_streme(
        f"{BASE_DIR}streme_out",
        os.path.join(BASE_DIR, 'positives.fasta'),
        minw=5,
        maxw=15,
        dna=dna
    )

    # Auto-search eps or use specified value
    if eps is None and auto_search_eps:
        print("8. Auto-searching for best eps parameter...")
        best_eps, cluster_analysis_df = auto_search_best_eps(
            aligned_seqs, seq_scores, eps_search_range, eps_step
        )
    else:
        if eps is None:
            eps = 0.3
        best_eps = eps
        cluster_analysis_df = pd.DataFrame([{
            'eps': eps,
            'n_clusters': 0,
            'n_noise': 0
        }])
        print(f"8. Using specified eps: {eps}")

    return aligned_seqs, seq_scores, best_eps, cluster_analysis_df


def run_pipeline_part2(aligned_short_seqs, scores, eps, BASE_DIR, data_type, RM):
    """
    Run Part 2 of the pipeline: Consensus motif computation and validation.

    Args:
        aligned_short_seqs: Aligned short sequences
        scores: IG scores
        eps: DBSCAN eps parameter
        BASE_DIR: Base directory for outputs
        data_type: Data type
        RM: RNA modification type

    Returns:
        consensus_motif, ig_scores, pwm_list, used_eps
    """
    print(f"\n{'='*60}")
    print(f"Running Part 2 for {RM}")
    print(f"{'='*60}")
    print(f"Using eps: {eps}")

    # Compute consensus motifs
    print("1. Computing consensus motifs...")
    consensus_motif, ig_scores, pwm_list, used_eps = compute_consensus_motifs(
        aligned_short_seqs, scores, eps, BASE_DIR, data_type, RM
    )

    print(f"Computed {len(pwm_list)} consensus motifs")

    # Export to MEME format for Tomtom
    print("2. Exporting to MEME format...")
    model_meme_path = os.path.join(BASE_DIR, 'model_motifs.meme')
    export_to_meme(pwm_list, model_meme_path)

    # Run Tomtom validation
    print("3. Running Tomtom validation...")
    streme_meme_path = os.path.join(BASE_DIR, 'streme_out', 'streme.txt')

    if os.path.exists(streme_meme_path):
        run_tomtom_validation(
            model_meme_path,
            streme_meme_path,
            f"{BASE_DIR}tomtom_out"
        )
    else:
        print(f"Warning: STREME output not found at {streme_meme_path}")

    return consensus_motif, ig_scores, pwm_list, used_eps


def run_full_pipeline(data_path, model_weights, data_type, RM, RM_index, length,
                      w, k, eps=None, auto_search_eps=False, eps_search_range=[0.1, 3],
                      eps_step=0.1, sample_size=5000, BASE_DIR=None,
                      alignment_script=None, seed=666, batch_size=128, dna=False):
    """
    Run the complete pipeline end-to-end.

    Args:
        data_path: Path to HDF5 data file
        model_weights: Path to model weights
        data_type: 'train', 'valid', or 'test'
        RM: RNA modification type name
        RM_index: Index of RNA modification
        length: Sequence length
        w: Window size for IG aggregation
        k: Top-K windows to select
        eps: Fixed eps value
        auto_search_eps: Whether to auto-search for best eps
        eps_search_range: Range for eps search
        eps_step: Step size for eps search
        sample_size: Number of samples to process
        BASE_DIR: Base directory for outputs
        alignment_script: Path to alignment script
        seed: Random seed for sample selection (default: 666)
        batch_size: Batch size for model inference (default: 128)
        dna: If True, pass --dna flag to STREME (treats sequences as DNA with ACGT alphabet)

    Returns:
        consensus_motif, ig_scores, pwm_list, used_eps
    """
    # Part 1
    aligned_seqs, scores, best_eps, cluster_df = run_pipeline_part1(
        data_path, model_weights, data_type, RM, RM_index, length,
        w, k, auto_search_eps, eps_search_range, eps_step, sample_size,
        BASE_DIR, alignment_script, eps, seed, batch_size, dna
    )

    # Part 2
    consensus_motif, ig_scores, pwm_list, used_eps = run_pipeline_part2(
        aligned_seqs, scores, best_eps if eps is None else eps,
        BASE_DIR, data_type, RM
    )

    return consensus_motif, ig_scores, pwm_list, used_eps
