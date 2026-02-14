import pandas as pd
import numpy as np


def read_h5_with_pandas(file_path):
    """
    使用 pandas 读取 HDF5 文件
    
    Args:
        file_path (str): HDF5 文件路径
    
    Returns:
        dict: 包含所有数据集的字典，键为数据集名称，值为 DataFrame
    """
    print(f"正在使用 pandas 读取 HDF5 文件: {file_path}")
    print("="*50)
    
    data = {}
    
    # 读取所有可用的数据集
    with pd.HDFStore(file_path, mode='r') as store:
        print("可用的数据集:")
        print(store.keys())
        print("\n")
        
        # 读取每个数据集
        for key in store.keys():
            # 去掉开头的 '/'
            df_name = key.lstrip('/')
            print(f"读取数据集: {df_name}")
            print("-" * 40)
            
            df = store.get(key)
            data[df_name] = df
            
            print(f"形状: {df.shape}")
            print(f"列名: {list(df.columns)[:5]}..." if len(df.columns) > 5 else f"列名: {list(df.columns)}")
            print(f"数据类型:\n{df.dtypes[:5]}\n" if len(df.dtypes) > 5 else f"数据类型:\n{df.dtypes}\n")
            
            # 显示前几行数据
            print("前5行数据:")
            print(df.head())
            print("\n")
    
    print("="*50)
    return data


def read_specific_dataset(file_path, dataset_name):
    """
    读取 HDF5 文件中的特定数据集
    
    Args:
        file_path (str): HDF5 文件路径
        dataset_name (str): 数据集名称（如 'train_in_nucleo'）
    
    Returns:
        DataFrame: 读取的数据
    """
    print(f"读取数据集: {dataset_name}")
    print("="*50)
    
    # 使用 read_hdf 直接读取
    df = pd.read_hdf(file_path, key=dataset_name)
    
    print(f"形状: {df.shape}")
    print(f"索引类型: {type(df.index)}")
    print(f"列数: {len(df.columns)}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n统计信息:\n{df.describe()}")
    print(f"\n前5行数据:\n{df.head()}")
    
    print("="*50)
    return df


def read_all_datasets_separately(file_path):
    """
    分别读取所有数据集并返回字典
    
    Args:
        file_path (str): HDF5 文件路径
    
    Returns:
        dict: 包含所有数据集的字典
    """
    datasets = ['test_in_nucleo', 'test_out', 'train_in_nucleo', 
                'train_out', 'valid_in_nucleo', 'valid_out']
    
    all_data = {}
    
    print("分别读取所有数据集:")
    print("="*50)
    
    for dataset_name in datasets:
        try:
            df = pd.read_hdf(file_path, key=dataset_name)
            all_data[dataset_name] = df
            print(f"✓ {dataset_name}: {df.shape}")
        except Exception as e:
            print(f"✗ {dataset_name}: 读取失败 - {e}")
    
    print("="*50)
    return all_data


def analyze_datasets(data_dict):
    """
    分析读取到的数据集
    
    Args:
        data_dict (dict): 包含所有 DataFrame 的字典
    """
    print("\n数据集分析:")
    print("="*50)
    
    for name, df in data_dict.items():
        print(f"\n数据集: {name}")
        print(f"  形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")
        print(f"  内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 检查是否有缺失值
        missing = df.isnull().sum().sum()
        print(f"  缺失值数量: {missing}")
        
        # 显示列名的示例
        print(f"  前3个列名: {list(df.columns)[:3]}")
    
    print("="*50)


if __name__ == "__main__":
    file_path = "MultiRM_data.h5"
    
    print("\n方法 1: 读取所有数据集")
    print("-" * 50)
    data = read_h5_with_pandas(file_path)
    
    print("\n方法 2: 读取特定数据集 (以 train_out 为例)")
    print("-" * 50)
    train_out = read_specific_dataset(file_path, 'train_out')
    
    print("\n方法 3: 分别读取所有数据集到字典")
    print("-" * 50)
    all_datasets = read_all_datasets_separately(file_path)
    
    # 分析数据集
    analyze_datasets(all_datasets)
    
    # 使用示例
    print("\n使用示例:")
    print("="*50)
    print("访问训练集输出数据:")
    print(f"train_out 的形状: {train_out.shape}")
    print(f"train_out 的列名: {list(train_out.columns)}")
    print(f"前3行的第一列:")
    print(train_out.iloc[:3, 0])
    print("="*50)
