import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE
from typing import Union, List
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_node_features(filepath: str) -> np.ndarray:
    """
    从文本文件中加载节点特征。
    假设文件中的每一行代表一个节点，特征值由空格或制表符分隔。

    Args:
        filepath (str): 特征文件的路径。

    Returns:
        np.ndarray: 一个形状为 (num_nodes, num_features) 的NumPy数组。
    """
    try:
        print(f"正在从 '{filepath}' 加载节点特征...")
        # 使用 np.loadtxt 可以高效地读取纯数字的文本文件
        features = np.loadtxt(filepath)
        print(f"加载成功！特征矩阵形状: {features.shape}")
        return features
    except FileNotFoundError:
        print(f"错误: 找不到特征文件 '{filepath}'。请检查路径。")
        return None
    except Exception as e:
        print(f"加载特征文件时发生错误: {e}")
        return None

def visualize_clustering_with_tsne(
    node_features: np.ndarray, # <--- 修改这里，直接接收矩阵
    Z_true: Union[List, np.ndarray],
    Z_pred: Union[List, np.ndarray],
    perplexity: int = 30,
    random_state: int = 42
):
    """
    使用t-SNE对节点特征进行降维，并可视化真实标签和预测标签的聚类结果。
    """
    # 步骤 1: 不再需要加载文件，直接检查输入
    if node_features is None:
        print("错误：输入的节点特征为 None。")
        return
    # 步骤 2: 计算t-SNE二维嵌入
    # 这个过程可能需要一些时间，具体取决于节点数量
    print("正在计算t-SNE二维嵌入... (这可能需要几分钟)")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_jobs=-1  # 使用所有可用的CPU核心
    )
    embedding = tsne.fit(node_features)
    print("t-SNE计算完成。")

    # 步骤 3: 绘制两个并排的图
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 绘制真实标签图
    _plot_scatter(axes[0], embedding, Z_true, "t-SNE Visualization (Ground Truth)")
    
    # 绘制预测标签图
    _plot_scatter(axes[1], embedding, Z_pred, "t-SNE Visualization (Predicted Clusters)")

    fig.suptitle("Clustering Result Visualization via t-SNE", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def _plot_scatter(ax, embedding, labels, title):
    """一个用于绘制单个散点图的辅助函数。"""
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    
    # 使用一个颜色映射表来自动为不同数量的簇生成颜色
    # 'tab20' 是一个很好的选择，因为它为20个不同的类别提供了独特的颜色
    colors = plt.cm.get_cmap('tab20', num_labels)
    
    for i, label in enumerate(unique_labels):
        # 找到属于当前标签的所有点的索引
        indices = np.where(labels == label)[0]
        
        # 绘制这些点
        ax.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            color=colors(i),
            label=f'Class {label}',
            alpha=0.7, # 使用一点透明度，以防点重叠
            s=20 # 设置点的大小
        )
        
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(title="Labels", markerscale=1.5)
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_confusion_matrix(true_labels, pred_labels, title, class_names):
    """
    计算并绘制混淆矩阵的热力图。

    Args:
        true_labels (np.ndarray): 真实的标签数组。
        pred_labels (np.ndarray): 模型预测的标签数组。
        title (str): 图表的标题。
        class_names (list): 类别名称列表，用于标记坐标轴。
    """
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 使用Seaborn绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # 显示图表
    plt.show()