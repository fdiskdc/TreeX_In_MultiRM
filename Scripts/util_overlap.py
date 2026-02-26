"""
工具函数库：用于分析MultiRM和TreeX模型在motif发现上的重叠度

Author: Cline
Date: 2026-02-24
Description: 提供函数用于计算两个模型权重Top-5位置的重叠度，并进行统计检验
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hypergeom
from typing import Tuple, List, Dict


def extract_top_windows_and_positions(
    attribution: np.ndarray,
    w: int = 12,
    k: int = 2,
    top_k: int = 5
) -> Tuple[List[int], List[int]]:
    """
    从IG attribution中提取Top-k窗口，然后从这些窗口中提取Top-5权重最高的碱基位置
    
    Parameters:
    -----------
    attribution : np.ndarray
        IG attribution值，形状为 (seq_len,) 或 (4, seq_len)
    w : int
        窗口大小
    k : int
        选择的窗口数量
    top_k : int
        从窗口中提取的Top-k碱基位置数量
        
    Returns:
    --------
    Tuple[List[int], List[int]]
        (window_positions, top_positions)
        - window_positions: top-k窗口的中心位置列表
        - top_positions: 从窗口中提取的top-k碱基位置列表
    """
    # 如果attribution是(4, seq_len)格式，转换为(seq_len,)
    if attribution.ndim == 2 and attribution.shape[0] == 4:
        attribution = np.mean(attribution, axis=0)
    
    seq_len = len(attribution)
    
    # 计算每个窗口的总分（滑动窗口）
    window_scores = []
    window_centers = []
    for i in range(seq_len - w + 1):
        window_score = np.sum(np.abs(attribution[i:i+w]))
        window_scores.append(window_score)
        window_centers.append(i + w // 2)
    
    window_scores = np.array(window_scores)
    window_centers = np.array(window_centers)
    
    # 选择Top-k窗口
    top_window_indices = np.argsort(window_scores)[-k:][::-1]
    selected_windows = []
    for idx in top_window_indices:
        center = int(window_centers[idx])
        start = max(0, center - w // 2)
        end = min(seq_len, center + w // 2)
        selected_windows.append((int(start), int(end)))
    
    # 合并所有窗口内的位置
    all_positions = set()
    for start, end in selected_windows:
        all_positions.update(range(start, end))
    all_positions = list(all_positions)
    
    # 从这些位置中选择Top-k权重最高的位置
    if len(all_positions) > 0:
        position_weights = {pos: abs(attribution[pos]) for pos in all_positions}
        top_positions = sorted(position_weights.items(), key=lambda x: x[1], reverse=True)
        top_positions = [pos for pos, _ in top_positions[:top_k]]
    else:
        # 如果没有窗口（极端情况），直接从整个序列选Top-k
        top_indices = np.argsort(np.abs(attribution))[-top_k:][::-1]
        top_positions = list(top_indices)
    
    window_centers_list = [int(window_centers[idx]) for idx in top_window_indices]

    return window_centers_list, top_positions


def calculate_overlap_stats(
    positions1: List[int],
    positions2: List[int],
    seq_len: int = 51
) -> Dict:
    """
    计算两个位置集合的重叠统计量
    
    Parameters:
    -----------
    positions1, positions2 : List[int]
        两个位置集合
    seq_len : int
        序列总长度
        
    Returns:
    --------
    Dict
        包含重叠统计的字典
    """
    set1 = set(positions1)
    set2 = set(positions2)
    
    intersection = set1 & set2
    overlap_size = len(intersection)
    
    # 超几何检验参数
    # M: 总体大小 (seq_len)
    # N: 第一个集合的大小 (Top-5 = 5)
    # n: 第二个集合的大小 (Top-5 = 5)
    # k: 实际重叠大小
    M = seq_len
    N = len(set1)
    n = len(set2)
    
    # 计算P值（右尾检验：观察到至少k个重叠的概率）
    p_value = hypergeom.sf(overlap_size - 1, M=M, N=N, n=n)
    
    # 期望重叠（随机情况）
    expected_overlap = (N * n) / M
    
    return {
        'overlap_size': overlap_size,
        'intersection': list(intersection),
        'set1_size': N,
        'set2_size': n,
        'expected_overlap': expected_overlap,
        'p_value': p_value
    }


def compute_all_overlaps(
    multirm_positions_list: List[List[int]],
    treex_positions_list: List[List[int]],
    seq_len: int = 51
) -> Dict:
    """
    汇总所有序列的重叠数据，计算整体统计量
    
    Parameters:
    -----------
    multirm_positions_list : List[List[int]]
        MultiRM为每个样本提取的Top-5位置列表
    treex_positions_list : List[List[int]]
        TreeX为每个样本提取的Top-5位置列表
    seq_len : int
        序列总长度
        
    Returns:
    --------
    Dict
        整体统计结果
    """
    n_samples = len(multirm_positions_list)
    
    # 计算每条序列的重叠
    overlap_sizes = []
    p_values = []
    all_intersections = []
    
    for i in range(n_samples):
        stats = calculate_overlap_stats(
            multirm_positions_list[i],
            treex_positions_list[i],
            seq_len
        )
        overlap_sizes.append(stats['overlap_size'])
        p_values.append(stats['p_value'])
        all_intersections.append(stats['intersection'])
    
    # 汇总统计
    total_overlaps = sum(overlap_sizes)
    avg_overlap = np.mean(overlap_sizes)
    std_overlap = np.std(overlap_sizes)
    
    # 期望平均重叠（随机情况）
    expected_avg = (5 * 5) / seq_len
    
    # 整体统计检验
    # 使用二项分布检验：将每条序列看作一次试验
    # 成功概率 = expected_overlap / 5 (归一化到0-1之间)
    # 但这里我们直接用超几何检验的总和版本
    
    # 计算观察到的总重叠与期望总重叠的差异显著性
    # 使用正态近似（因为样本数通常较大）
    # Z = (total_overlaps - expected_total) / std_total
    
    # 另一种方法：使用t检验比较平均重叠与期望平均重叠
    from scipy.stats import ttest_1samp
    t_stat, t_p_value = ttest_1samp(overlap_sizes, popmean=expected_avg)
    
    # Fisher方法合并P值
    from scipy.stats import combine_pvalues
    fisher_stat, fisher_p_value = combine_pvalues(p_values, method='fisher')
    
    return {
        'n_samples': n_samples,
        'overlap_sizes': overlap_sizes,
        'total_overlaps': total_overlaps,
        'avg_overlap': avg_overlap,
        'std_overlap': std_overlap,
        'expected_avg_overlap': expected_avg,
        'individual_p_values': p_values,
        't_stat': t_stat,
        't_p_value': t_p_value,
        'fisher_p_value': fisher_p_value,
        'all_intersections': all_intersections
    }


def plot_overlap_distribution(
    overlap_sizes: List[int],
    expected_overlap: float,
    title: str = "Overlap Distribution",
    save_path: str = None
):
    """
    绘制重叠度分布直方图
    
    Parameters:
    -----------
    overlap_sizes : List[int]
        每条序列的交集大小列表
    expected_overlap : float
        期望重叠大小（随机情况）
    title : str
        图表标题
    save_path : str, optional
        保存路径
    """
    plt.figure(figsize=(8, 5))
    
    # 绘制直方图
    counts = [overlap_sizes.count(i) for i in range(6)]
    bars = plt.bar(range(6), counts, color='steelblue', alpha=0.7, edgecolor='black')
    
    # 标记期望值
    plt.axvline(x=expected_overlap, color='red', linestyle='--', linewidth=2, 
                label=f'Expected (random) = {expected_overlap:.2f}')
    
    # 标记平均值
    avg_overlap = np.mean(overlap_sizes)
    plt.axvline(x=avg_overlap, color='green', linestyle='-', linewidth=2,
                label=f'Observed avg = {avg_overlap:.2f}')
    
    # 在每个柱子上标注数量
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Overlap Size (Top-5)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved overlap distribution plot to: {save_path}")
    
    plt.show()


def plot_position_heatmap(
    multirm_weights: np.ndarray,
    treex_weights: np.ndarray,
    sample_idx: int = 0,
    title: str = "Position Weight Comparison",
    save_path: str = None
):
    """
    绘制两个模型在序列位置上的权重热图
    
    Parameters:
    -----------
    multirm_weights : np.ndarray
        MultiRM的权重矩阵，形状为 (n_samples, seq_len) 或 (4, seq_len)
    treex_weights : np.ndarray
        TreeX的权重矩阵
    sample_idx : int
        样本索引（如果是多个样本）
    title : str
        图表标题
    save_path : str, optional
        保存路径
    """
    # 提取单个样本的权重
    if multirm_weights.ndim == 2:
        multirm_weight = multirm_weights[sample_idx]
    else:
        multirm_weight = multirm_weights
    
    if treex_weights.ndim == 2:
        treex_weight = treex_weights[sample_idx]
    else:
        treex_weight = treex_weights
    
    # 如果是(4, seq_len)格式，转换为(seq_len,)
    if multirm_weight.ndim == 2:
        multirm_weight = np.mean(multirm_weight, axis=0)
    if treex_weight.ndim == 2:
        treex_weight = np.mean(treex_weight, axis=0)
    
    seq_len = len(multirm_weight)
    positions = np.arange(seq_len)
    
    # 创建热图数据
    heatmap_data = np.array([multirm_weight, treex_weight])
    
    plt.figure(figsize=(12, 3))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0,
                xticklabels=positions,
                yticklabels=['MultiRM', 'TreeX'],
                cbar_kws={'label': 'Weight Value'},
                linewidths=0.5)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved position heatmap to: {save_path}")
    
    plt.show()


def create_statistics_table(
    stats: Dict,
    rm_name: str,
    save_path: str = None
) -> pd.DataFrame:
    """
    创建统计表格
    
    Parameters:
    -----------
    stats : Dict
        从compute_all_overlaps返回的统计结果
    rm_name : str
        修饰名称
    save_path : str, optional
        保存路径（CSV格式）
        
    Returns:
    --------
    pd.DataFrame
        统计表格
    """
    # 准备表格数据
    data = {
        'Metric': [
            'Total Samples',
            'Average Overlap',
            'Std Overlap',
            'Expected Overlap (random)',
            'Total Overlaps',
            'Overlap Ratio (Observed/Expected)',
            'T-test Statistic',
            'T-test P-value',
            'Fisher P-value (combined)'
        ],
        'Value': [
            stats['n_samples'],
            f"{stats['avg_overlap']:.4f}",
            f"{stats['std_overlap']:.4f}",
            f"{stats['expected_avg_overlap']:.4f}",
            stats['total_overlaps'],
            f"{stats['avg_overlap'] / stats['expected_avg_overlap']:.4f}" if stats['expected_avg_overlap'] > 0 else "N/A",
            f"{stats['t_stat']:.4f}",
            f"{stats['t_p_value']:.2e}",
            f"{stats['fisher_p_value']:.2e}"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # 保存到文件
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved statistics table to: {save_path}")
    
    return df


def analyze_single_rm(
    multirm_model,
    treex_model,
    X: np.ndarray,
    y: np.ndarray,
    rm_index: int,
    rm_name: str,
    w: int = 12,
    k: int = 2,
    top_k: int = 5,
    seq_len: int = 51,
    output_dir: str = None,
    plot_individual_heatmaps: bool = False,
    max_heatmaps: int = 5
) -> Dict:
    """
    分析单个修饰类型的重叠度
    
    Parameters:
    -----------
    multirm_model, treex_model : Model
        两个模型
    X, y : np.ndarray
        数据和标签
    rm_index : int
        修饰索引
    rm_name : str
        修饰名称
    w, k, top_k, seq_len : int
        参数
    output_dir : str, optional
        输出目录
    plot_individual_heatmaps : bool
        是否绘制单个样本的热图
    max_heatmaps : int
        最多绘制多少个热图
        
    Returns:
    --------
    Dict
        分析结果
    """
    from util_ig import random_baseline_integrated_gradients, calculate_outputs_and_gradients
    
    # 筛选阳性样本
    true_labels = y[:, rm_index].cpu().numpy() if hasattr(y, 'numpy') else y[:, rm_index].numpy()
    positive_indices = np.where(true_labels == 1)[0]
    
    print(f"\n{'='*60}")
    print(f"Analyzing {rm_name} (index {rm_index})")
    print(f"Total positive samples: {len(positive_indices)}")
    print(f"{'='*60}")
    
    # 准备数据
    X_pos = X[positive_indices]
    
    # 计算MultiRM的IG权重
    print("Computing MultiRM IG weights...")
    multirm_attributions = []
    for idx in range(len(positive_indices)):
        # 假设MultiRM模型直接输出logits
        attribution = random_baseline_integrated_gradients(
            torch.unsqueeze(X_pos[idx], 0),
            multirm_model,
            calculate_outputs_and_gradients,
            index=rm_index,
            steps=50,
            num_random_trials=10,
            cuda=None
        )
        multirm_attributions.append(attribution)
    multirm_attributions = np.array(multirm_attributions)
    
    # 计算TreeX的IG权重
    print("Computing TreeX IG weights...")
    treex_attributions = []
    for idx in range(len(positive_indices)):
        # TreeX模型返回 (logits, attn_weights)，需要特殊处理
        attribution = random_baseline_integrated_gradients(
            torch.unsqueeze(X_pos[idx], 0),
            treex_model,
            calculate_outputs_and_gradients,
            index=rm_index,
            steps=50,
            num_random_trials=10,
            cuda=None
        )
        treex_attributions.append(attribution)
    treex_attributions = np.array(treex_attributions)
    
    # 提取Top-5位置
    print("Extracting Top-5 positions...")
    multirm_positions_list = []
    treex_positions_list = []
    
    for i in range(len(positive_indices)):
        _, multirm_pos = extract_top_windows_and_positions(
            multirm_attributions[i], w=w, k=k, top_k=top_k
        )
        _, treex_pos = extract_top_windows_and_positions(
            treex_attributions[i], w=w, k=k, top_k=top_k
        )
        multirm_positions_list.append(multirm_pos)
        treex_positions_list.append(treex_pos)
    
    # 计算重叠统计
    print("Computing overlap statistics...")
    stats = compute_all_overlaps(multirm_positions_list, treex_positions_list, seq_len)
    
    # 打印结果
    print(f"\nResults for {rm_name}:")
    print(f"  Total samples: {stats['n_samples']}")
    print(f"  Average overlap: {stats['avg_overlap']:.4f} ± {stats['std_overlap']:.4f}")
    print(f"  Expected overlap (random): {stats['expected_avg_overlap']:.4f}")
    print(f"  Overlap ratio: {stats['avg_overlap'] / stats['expected_avg_overlap']:.2f}x")
    print(f"  T-test P-value: {stats['t_p_value']:.2e}")
    print(f"  Fisher P-value: {stats['fisher_p_value']:.2e}")
    
    # 保存结果
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 重叠度分布直方图
        plot_overlap_distribution(
            stats['overlap_sizes'],
            stats['expected_avg_overlap'],
            title=f"{rm_name}: Overlap Distribution (w={w}, k={k})",
            save_path=os.path.join(output_dir, f"{rm_name}_overlap_distribution.png")
        )
        
        # 2. 位置热图（选取前几个样本）
        if plot_individual_heatmaps:
            for i in range(min(max_heatmaps, len(positive_indices))):
                plot_position_heatmap(
                    multirm_attributions[i],
                    treex_attributions[i],
                    sample_idx=0,
                    title=f"{rm_name} - Sample {i}: Position Weights",
                    save_path=os.path.join(output_dir, f"{rm_name}_heatmap_sample_{i}.png")
                )
        
        # 3. 统计表格
        stats_df = create_statistics_table(
            stats,
            rm_name,
            save_path=os.path.join(output_dir, f"{rm_name}_statistics.csv")
        )
        
        print(f"\nResults saved to: {output_dir}")
    
    return {
        'rm_name': rm_name,
        'stats': stats,
        'multirm_attributions': multirm_attributions,
        'treex_attributions': treex_attributions,
        'multirm_positions': multirm_positions_list,
        'treex_positions': treex_positions_list,
        'positive_indices': positive_indices
    }