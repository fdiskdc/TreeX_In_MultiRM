import numpy as np
import pandas as pd

def extract_zero_label_sequences():
    """
    Extract RNA sequences with all-zero labels from MultiRM_data.h5
    and save them to zero_seq.npy and zero_label*.npy files.
    """
    
    # Initialize lists to store data from all splits
    all_sequences = []
    all_labels = []
    
    # Read HDF5 file
    print("Reading MultiRM_data.h5...")
    
    # Process test, train, and valid datasets
    for split in ['test', 'train', 'valid']:
        print(f"Processing {split} split...")
        
        # Read sequences (_in_nucleo)
        seq_key = f'{split}_in_nucleo'
        label_key = f'{split}_out'
        
        # Read sequences DataFrame
        seq_df = pd.read_hdf('MultiRM_data.h5', key=seq_key)
        # Read labels DataFrame
        label_df = pd.read_hdf('MultiRM_data.h5', key=label_key)
        
        print(f"  {split} sequences shape: {seq_df.shape}")
        print(f"  {split} labels shape: {label_df.shape}")
        
        # Convert DataFrame rows to strings (join characters along columns)
        seq_strings = [''.join(row) for row in seq_df.values]
        
        all_sequences.extend(seq_strings)
        all_labels.append(label_df.values)
    
    # Combine all labels
    all_labels = np.vstack(all_labels)
    all_sequences = np.array(all_sequences)
    
    print(f"\nTotal sequences: {all_sequences.shape[0]}")
    print(f"Total labels shape: {all_labels.shape}")
    
    # Find sequences with all-zero labels (all 12 columns are 0)
    # axis=1 means check if all columns in each row are 0
    zero_mask = np.all(all_labels == 0, axis=1)
    zero_count = np.sum(zero_mask)
    
    print(f"\nFound {zero_count} sequences with all-zero labels")
    
    # Extract zero-label sequences
    zero_sequences = all_sequences[zero_mask]
    
    print(f"zero_sequences shape: {zero_sequences.shape}")
    
    # Save zero_seq.npy - shape (n, 1001) string array with dtype S1 (bytes)
    # Convert each string to array of characters
    zero_seq_array = np.array([list(seq) for seq in zero_sequences], dtype='S1')
    print(f"zero_seq_array shape: {zero_seq_array.shape}, dtype: {zero_seq_array.dtype}")
    
    np.save('zero_seq.npy', zero_seq_array)
    print("Saved zero_seq.npy")
    
    # Save zero_label4.npy - shape (n, 4) all zeros
    n = len(zero_sequences)
    zero_label4 = np.zeros((n, 4), dtype=np.float64)
    np.save('zero_label4.npy', zero_label4)
    print("Saved zero_label4.npy")
    
    # Save zero_label12.npy - shape (n, 12) all zeros
    zero_label12 = np.zeros((n, 12), dtype=np.float64)
    np.save('zero_label12.npy', zero_label12)
    print("Saved zero_label12.npy")
    
    # Save zero_label1001.npy - shape (n, 1001) all zeros
    zero_label1001 = np.zeros((n, 1001), dtype=np.float64)
    np.save('zero_label1001.npy', zero_label1001)
    print("Saved zero_label1001.npy")
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"  zero_seq.npy: shape {zero_seq_array.shape}, dtype {zero_seq_array.dtype} - RNA sequences with zero labels")
    print(f"  zero_label4.npy: shape {zero_label4.shape} - all zeros")
    print(f"  zero_label12.npy: shape {zero_label12.shape} - all zeros")
    print(f"  zero_label1001.npy: shape {zero_label1001.shape} - all zeros")
    print("="*50)

if __name__ == '__main__':
    extract_zero_label_sequences()
