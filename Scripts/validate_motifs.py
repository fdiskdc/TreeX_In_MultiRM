#!/usr/bin/env python3
"""
validate_motifs.py

Script to validate TreeX model motifs against STREME motifs using Tomtom comparison.

This script implements three core modules:
1. export_to_meme: Export PWM matrices to MEME format
2. run_tomtom: Run Tomtom for automated motif comparison
3. parse_tomtom_results: Parse and output validation report
"""

import numpy as np
import os
import subprocess
import sys


def export_to_meme(pwm_list, output_filepath, motif_names=None, bg_frequencies=None):
    """
    Export PWM matrices to MEME format file.
    
    Args:
        pwm_list: List of numpy arrays, each with shape (L, 4) where L is motif length
                  and 4 corresponds to A, C, G, T. Values should be probabilities.
        output_filepath: Path to output MEME format file
        motif_names: Optional list of motif names. If None, generates "Model_Motif_N"
        bg_frequencies: Background frequencies for A, C, G, T. Default: A=0.295, C=0.205, G=0.205, T=0.295
    
    Raises:
        ValueError: If PWM rows don't sum to approximately 1.0
    """
    if bg_frequencies is None:
        bg_frequencies = {'A': 0.295, 'C': 0.205, 'G': 0.205, 'T': 0.295}
    
    if motif_names is None:
        motif_names = [f"Model_Motif_{i+1}" for i in range(len(pwm_list))]
    
    pseudocount = 0.001
    
    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_filepath, 'w') as f:
        f.write("MEME version 5.0\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("Background letter frequencies\n")
        f.write(f"A {bg_frequencies['A']} C {bg_frequencies['C']} G {bg_frequencies['G']} T {bg_frequencies['T']}\n\n")
        
        for idx, pwm in enumerate(pwm_list):
            pwm = np.array(pwm)
            
            if pwm.ndim != 2 or pwm.shape[1] != 4:
                raise ValueError(f"Motif {idx} has invalid shape {pwm.shape}. Expected (L, 4)")
            
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
    
    print(f"MEME file written to: {output_filepath}")


def run_tomtom(query_meme, target_meme, output_dir, min_overlap=1, dist_method='pearson'):
    """
    Run Tomtom motif comparison tool.
    
    Args:
        query_meme: Path to query MEME file (model motifs)
        target_meme: Path to target MEME file (STREME motifs)
        output_dir: Output directory for Tomtom results
        min_overlap: Minimum overlap between motifs (default: 1)
        dist_method: Distance method - 'pearson', 'spearman', 'ks', 'edit' (default: 'pearson')
    
    Returns:
        Path to Tomtom results TSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tomtom_cmd = [
        'tomtom',
        '-no-ssc',
        '-oc', output_dir,
        '-min-overlap', str(min_overlap),
        '-dist', dist_method,
        query_meme,
        target_meme
    ]
    
    print(f"Running Tomtom command: {' '.join(tomtom_cmd)}")
    
    try:
        result = subprocess.run(
            tomtom_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print("Tomtom completed successfully")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Tomtom error: {e.stderr}")
        raise
    except FileNotFoundError:
        print("Error: Tomtom not found. Please ensure MEME Suite is installed and in PATH")
        raise
    
    tsv_path = os.path.join(output_dir, 'tomtom.tsv')
    return tsv_path


def parse_tomtom_results(tomtom_tsv_path, evalue_threshold=0.05):
    """
    Parse Tomtom results and print validation report.
    
    Args:
        tomtom_tsv_path: Path to Tomtom TSV output file
        evalue_threshold: E-value threshold for significant matches (default: 0.05)
    
    Returns:
        List of significant matches as dictionaries
    """
    if not os.path.exists(tomtom_tsv_path):
        raise FileNotFoundError(f"Tomtom results not found at: {tomtom_tsv_path}")
    
    significant_matches = []
    
    with open(tomtom_tsv_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print("No Tomtom results found")
        return significant_matches
    
    header = lines[0].strip().split('\t')
    
    print("\n" + "=" * 80)
    print("TREE-X MOTIF VALIDATION REPORT")
    print("=" * 80)
    print(f"\nSignificant matches (E-value < {evalue_threshold}):")
    print("-" * 80)
    
    target_motifs_of_interest = ['1-AGYGAG', '2-CCAGCCTGGS']
    
    for line in lines[1:]:
        fields = line.strip().split('\t')
        
        if len(fields) < 4:
            continue
        
        try:
            query_id = fields[0]
            target_id = fields[1]
            evalue = float(fields[3])
            match_score = float(fields[4]) if len(fields) > 4 else 0.0
            overlap = fields[5] if len(fields) > 5 else 'N/A'
            
            if evalue < evalue_threshold:
                match_info = {
                    'query': query_id,
                    'target': target_id,
                    'evalue': evalue,
                    'score': match_score,
                    'overlap': overlap
                }
                significant_matches.append(match_info)
                
                is_target_of_interest = any(tm in target_id for tm in target_motifs_of_interest)
                highlight = " *** " if is_target_of_interest else " "
                
                print(f"{highlight}Query: {query_id:20s} -> Target: {target_id:20s} "
                      f"E-value: {evalue:.2e}  Score: {match_score:.4f}  Overlap: {overlap}")
                
        except (ValueError, IndexError):
            continue
    
    print("-" * 80)
    
    if significant_matches:
        target_ids = [m['target'] for m in significant_matches]
        
        motif1_found = any('1-AGYGAG' in t for t in target_ids)
        motif2_found = any('2-CCAGCCTGGS' in t for t in target_ids)
        
        print("\nSUMMARY:")
        print(f"  Total significant matches: {len(significant_matches)}")
        print(f"  MOTIF 1-AGYGAG matched: {'YES' if motif1_found else 'NO'}")
        print(f"  MOTIF 2-CCAGCCTGGS matched: {'YES' if motif2_found else 'NO'}")
        
        if motif1_found or motif2_found:
            print("\n*** VALIDATION SUCCESSFUL ***")
            print("TreeX model motifs show significant similarity to STREME reference motifs.")
        else:
            print("\n*** VALIDATION INCONCLUSIVE ***")
            print("No match found to the top two STREME motifs (1-AGYGAG and 2-CCAGCCTGGS).")
    else:
        print("\nNo significant matches found (E-value >= 0.05)")
        print("*** VALIDATION FAILED ***")
        print("TreeX model motifs do not show significant similarity to STREME reference motifs.")
    
    print("=" * 80 + "\n")
    
    return significant_matches


def validate_motifs(pwm_list, streme_meme_path, output_dir='./tomtom_output', 
                    motif_names=None, evalue_threshold=0.05):
    """
    Main function to run the complete validation pipeline.
    
    Args:
        pwm_list: List of PWM matrices from TreeX model (each shape: L x 4)
        streme_meme_path: Path to STREME MEME file
        output_dir: Output directory for results
        motif_names: Optional list of motif names
        evalue_threshold: E-value threshold for significant matches
    
    Returns:
        List of significant matches
    """
    query_meme_path = os.path.join(output_dir, 'model_motifs.meme')
    
    print("Step 1: Exporting model PWM matrices to MEME format...")
    export_to_meme(pwm_list, query_meme_path, motif_names=motif_names)
    
    print("\nStep 2: Running Tomtom comparison...")
    tsv_path = run_tomtom(query_meme_path, streme_meme_path, output_dir)
    
    print("\nStep 3: Parsing Tomtom results...")
    matches = parse_tomtom_results(tsv_path, evalue_threshold=evalue_threshold)
    
    return matches


def load_consensus_motifs_from_notebook():
    """
    Placeholder function to load consensus motifs from notebook execution.
    
    In practice, you would run the notebook and extract the consensus_motif variable.
    The consensus_motif from cal_consensus_motif_2 has shape (n_clusters, 4, length)
    and needs to be transposed to (length, 4) for each motif.
    
    Returns:
        List of numpy arrays with shape (L, 4) for each motif
    """
    print("Note: This is a placeholder. To use:")
    print("1. Run consensus_motifs_cp1_treex.ipynb to generate consensus_motif")
    print("2. Extract the consensus_motif variable (shape: n_clusters x 4 x length)")
    print("3. Transpose each motif from (4, L) to (L, 4) using: motif.T")
    print("4. Pass the transposed motifs to validate_motifs()")
    return []


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate TreeX model motifs against STREME motifs using Tomtom'
    )
    parser.add_argument(
        '--pwm-npz', 
        type=str, 
        default='/home/dc/vscode/vscode20260112/MultiRM/fasta/treex_consensus_motifs.npz',
        help='Path to NPZ file containing PWM matrices'
    )
    parser.add_argument(
        '--streme', 
        type=str, 
        default='/home/dc/vscode/vscode20260112/MultiRM/fasta2/streme.txt',
        help='Path to STREME MEME file (default: fasta2/streme.txt)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./tomtom_output',
        help='Output directory for Tomtom results (default: ./tomtom_output)'
    )
    parser.add_argument(
        '--evalue-threshold',
        type=float,
        default=0.05,
        help='E-value threshold for significant matches (default: 0.05)'
    )
    
    args = parser.parse_args()
    
    if args.pwm_npz:
        print(f"Loading PWM matrices from: {args.pwm_npz}")
        data = np.load(args.pwm_npz, allow_pickle=True)
        
        if 'pwms' in data:
            pwm_list = data['pwms']
        elif 'motifs' in data:
            pwm_list = data['motifs']
        else:
            pwm_list = [data[key] for key in data.keys()]
        
        streme_path = args.streme
        if not os.path.isabs(streme_path):
            streme_path = os.path.join(os.path.dirname(__file__), streme_path)
        
        validate_motifs(
            pwm_list=pwm_list,
            streme_meme_path=streme_path,
            output_dir=args.output_dir,
            evalue_threshold=args.evalue_threshold
        )
    else:
        print("Usage:")
        print("  python validate_motifs.py --pwm-npz motifs.npz --streme fasta2/streme.txt")
        print("\nTo generate PWM matrices:")
        print("  1. Run consensus_motifs_cp1_treex.ipynb")
        print("  2. Extract consensus_motif from the notebook")
        print("  3. Save as NPZ: np.savez('motifs.npz', pwms=consensus_motif)")
        print("\nOr use validate_motifs() function directly in Python:")
        print("  from validate_motifs import validate_motifs")
        print("  # pwm_list: list of numpy arrays with shape (L, 4)")
        print("  matches = validate_motifs(pwm_list, 'fasta2/streme.txt')")
