import h5py
import numpy as np


def explore_h5_structure(file_path):
    """
    探查 HDF5 文件的数据结构
    
    Args:
        file_path (str): HDF5 文件路径
    """
    print(f"正在探查 HDF5 文件: {file_path}")
    print("="*50)
    
    with h5py.File(file_path, 'r') as f:
        # 递归打印 HDF5 文件结构
        def print_structure(name, obj, indent=0):
            spaces = "  " * indent
            if isinstance(obj, h5py.Dataset):
                print(f"{spaces}数据集: {name}")
                print(f"{spaces}  形状: {obj.shape}")
                print(f"{spaces}  数据类型: {obj.dtype}")
                print(f"{spaces}  大小: {obj.size}")
                # 显示部分数据样本
                try:
                    if obj.size > 0:
                        if len(obj.shape) == 1:
                            sample = obj[:5] if obj.shape[0] >= 5 else obj[:]
                        elif len(obj.shape) == 2:
                            sample = obj[:3, :5] if obj.shape[0] >= 3 and obj.shape[1] >= 5 else obj[:min(3,obj.shape[0]), :min(5,obj.shape[1])]
                        elif len(obj.shape) == 3:
                            sample = obj[:2, :3, :5] if obj.shape[0] >= 2 and obj.shape[1] >= 3 and obj.shape[2] >= 5 else obj[:min(2,obj.shape[0]), :min(3,obj.shape[1]), :min(5,obj.shape[2])]
                        else:
                            sample = obj  # 对于更高维度的数组，简单显示
                        print(f"{spaces}  数据样本: {sample}")
                except Exception as e:
                    print(f"{spaces}  数据样本: 无法读取 - {e}")
            elif isinstance(obj, h5py.Group):
                print(f"{spaces}组: {name}")
            
            # 打印属性
            if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                for key, val in obj.attrs.items():
                    print(f"{spaces}  属性: {key} = {val}")
        
        # 递归遍历整个文件结构
        f.visititems(lambda name, obj: print_structure(name, obj))
        
    print("="*50)


def get_basic_info(file_path):
    """
    获取 HDF5 文件的基本信息
    
    Args:
        file_path (str): HDF5 文件路径
    """
    print("基本文件信息:")
    with h5py.File(file_path, 'r') as f:
        print(f"  文件名: {file_path}")
        print(f"  根级键: {list(f.keys())}")
        print(f"  总数据集数量: {len(f)}")
        
        # 统计信息
        total_size = 0
        dataset_count = 0
        group_count = 0
        
        def count_items(name, obj):
            nonlocal total_size, dataset_count, group_count
            if isinstance(obj, h5py.Dataset):
                total_size += obj.size * obj.dtype.itemsize
                dataset_count += 1
            elif isinstance(obj, h5py.Group):
                group_count += 1
                
        f.visititems(count_items)
        
        print(f"  数据集总数: {dataset_count}")
        print(f"  组总数: {group_count}")
        print(f"  估算总大小: {total_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    file_path = "MultiRM_data.h5"
    
    # 获取基本信息
    get_basic_info(file_path)
    print()
    
    # 探查详细结构
    explore_h5_structure(file_path)
    
    # 额外分析 - 检查是否包含 RNA 修饰相关数据
    print("\n特定数据分析:")
    print("="*50)
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f"\n数据集 '{key}':")
                print(f"  形状: {item.shape}")
                print(f"  类型: {item.dtype}")
                
                # 如果是序列相关的数据，可以查看一些统计信息
                if item.dtype.kind in ['U', 'S']:  # Unicode 或字节字符串
                    print("  这是一个字符串数据集")
                    try:
                        # 尝试获取前几个值
                        values = item[:5] if item.shape[0] >= 5 else item[:]
                        print(f"  示例值: {values}")
                    except:
                        print("  无法读取示例值")
                        
                elif item.dtype in [np.int32, np.int64, np.float32, np.float64]:
                    # 数值数据，获取一些统计信息
                    data_slice = item[:5] if item.shape[0] >= 5 else item[:]
                    print(f"  示例值: {data_slice}")
                    if len(item.shape) == 1:
                        print(f"  最小值: {np.min(item)}")
                        print(f"  最大值: {np.max(item)}")
                        print(f"  平均值: {np.mean(item):.4f}")
            elif isinstance(item, h5py.Group):
                print(f"\n组 '{key}' 包含: {list(item.keys())}")