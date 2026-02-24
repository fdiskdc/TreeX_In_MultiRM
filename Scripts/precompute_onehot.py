
#!/usr/bin/env python3
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

def onehot_encode(nucleo_data):
    """
    Optimized one-hot encoding using list conversion for maximum speed
    
    Args:
        nucleo_data: pandas DataFrame containing nucleotide sequences
    
    Returns:
        pandas DataFrame containing one-hot encoded sequences
    """
    import numpy as np
    from tqdm import tqdm
    
    print(f"Processing {len(nucleo_data)} samples")
    
    # 转换为列表以提高访问速度
    print("Converting DataFrame to list...")
    data_list = nucleo_data.values.tolist()
    
    # 获取数据维度
    n_samples = len(data_list)
    n_features = len(data_list[0]) if n_samples > 0 else 0
    
    # 预分配结果数组
    encoded_array = np.zeros((n_samples, n_features * 4), dtype=np.int8)
    
    # 处理所有样本
    print("Encoding sequences...")
    for i in tqdm(range(n_samples), desc="Encoding samples"):
        row = data_list[i]
        for j, nuc in enumerate(row):
            pos = j * 4
            if nuc == 'A':
                encoded_array[i, pos] = 1
            elif nuc == 'C':
                encoded_array[i, pos + 1] = 1
            elif nuc == 'G':
                encoded_array[i, pos + 2] = 1
            elif nuc == 'T' or nuc == 'U':
                encoded_array[i, pos + 3] = 1
            # N remains 0
    
    print(f"Encoded shape: {encoded_array.shape}")
    return pd.DataFrame(encoded_array)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute one-hot encoding for MultiRM data")
    parser.add_argument('--input', default='MultiRM_data.h5', type=str, help="Input h5 file path")
    parser.add_argument('--output', default='MultiRM_data.h5', type=str, help="Output h5 file path")
    args = parser.parse_args()
    
    print(f"Reading data from {args.input}")
    
    # 读取nucleo数据
    train_in_nucleo = pd.read_hdf(args.input, 'train_in_nucleo')
    valid_in_nucleo = pd.read_hdf(args.input, 'valid_in_nucleo')
    test_in_nucleo = pd.read_hdf(args.input, 'test_in_nucleo')
    
    print("Processing train_in_nucleo")
    train_in = onehot_encode(train_in_nucleo)
    
    print("Processing valid_in_nucleo")
    valid_in = onehot_encode(valid_in_nucleo)
    
    print("Processing test_in_nucleo")
    test_in = onehot_encode(test_in_nucleo)
    
    print(f"Saving results to {args.output}")
    
    # 保存到h5文件
    train_in.to_hdf(args.output, key='train_in')
    valid_in.to_hdf(args.output, key='valid_in')
    test_in.to_hdf(args.output, key='test_in')
    
    print("Precomputation completed successfully!")

if __name__ == "__main__":
    main()
