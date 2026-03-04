# GNNESS: Graph Neural Network Guided Sparse Solver

This repository contains the implementation of **GNNESS**, a novel approach that leverages Graph Neural Networks (GNNs) to guide classical solvers for sparse polynomial interpolation and rank minimization problems. By predicting the underlying rank and stability of noisy Hankel matrices, GNNESS significantly improves the robustness and efficiency of parameter reconstruction.

## 🌟 Key Features

*   **RankGNN**: A specialized GNN architecture that processes Hankel graphs to predict the rank of the underlying linear recurrence relation.
*   **Hybrid Solver**: Combines GNN rank predictions with classical Sylvester-based solvers to achieve high accuracy even in the presence of noise.
*   **Meta-Solver**: An advanced decision-making module that uses a meta-classifier to choose the best reconstruction candidate (Neighbor, Top-3, or All strategies) based on GNN confidence, stability scores, and reconstruction residuals.
*   **Robustness**: Designed to handle noisy data where classical methods (like SVD truncation) often fail.

## 📂 Directory Structure

```
GNN-guided/
├── config.yaml                     # Main configuration file for model and training
├── data_generation.py              # Synthetic data generation (polynomials, Hankel matrices)
├── graph_builder.py                # Constructs Hankel graphs from coefficient sequences
├── models.py                       # PyTorch implementation of RankGNN and baselines
├── solver.py                       # Core solver logic (Classical, SVD, Hybrid, One-Shot)
├── train.py                        # Script to train the RankGNN model
├── evaluate.py                     # Evaluation scripts for all methods, including Meta-Solver
├── run_paper_experiments.py        # Script to reproduce the experiments from the paper
├── run_no_gnn_features_experiment.py # Experiment analyzing the impact of GNN features on Meta-Solver
├── utils.py                        # Utility functions
├── models/                         # Directory for saving/loading trained model weights
│   ├── rank_gnn_fast.pth
│   └── ...
└── train_stats.pt                  # Saved training statistics for feature normalization
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. The project relies on the following key libraries:

*   **PyTorch** (Core deep learning framework)
*   **PyTorch Geometric** (Graph neural network operations)
*   **NumPy & SciPy** (Numerical linear algebra)
*   **scikit-learn** (Metrics and meta-classifier)
*   **pandas** (Data manipulation for results)
*   **tqdm** (Progress bars)
*   **PyYAML** (Configuration management)

Install dependencies via pip:

```bash
pip install torch torch-geometric numpy scipy scikit-learn pandas tqdm pyyaml matplotlib
```

*(Note: Install PyTorch and PyTorch Geometric according to your CUDA version/OS from their official websites)*

### 🔧 Configuration

All hyperparameters are managed in `config.yaml`. Key sections include:

*   **`data`**: Sampling parameters (degree range `d_range`, rank range `r_range`, noise levels).
*   **`model`**: GNN architecture settings (`hidden_dim`, `layers`, `dropout`).
*   **`solver`**: Solver thresholds (`tau` for residual check).
*   **`training`**: Learning rate, batch size, epochs.

## 🏃 Usage

### 1. Training the Model

To train the RankGNN model from scratch:

```bash
python train.py
```

This will generate synthetic data on-the-fly, train the model, and save the best weights to `models/rank_gnn_fast.pth` (or as configured).

### 2. Evaluating Methods

To evaluate the performance of Classical, SVD, Hybrid, and Meta-Solver methods:

```bash
python evaluate.py
```

This script calculates accuracy, Valid Solution Rate (VSR), and runtime for various polynomial degrees.

### 3. Running Paper Experiments

To reproduce the tables and figures (Rank Accuracy, Runtime, Noise Robustness) discussed in the paper:

```bash
python run_paper_experiments.py
```

### 4. Meta-Solver Feature Analysis

To analyze the contribution of GNN-specific features (stability, confidence) to the Meta-Solver's performance:

```bash
python run_no_gnn_features_experiment.py
```

## 🧠 Method Overview

1.  **Input**: A noisy sequence of coefficients $a = (a_0, a_1, \dots, a_n)$.
2.  **Graph Construction**: The sequence is converted into a **Hankel Graph**, where nodes represent coefficients and edges represent Hankel matrix structure.
3.  **Rank Prediction**: **RankGNN** processes the graph to predict:
    *   **Rank ($r$)**: The number of exponential terms.
    *   **Stability Score**: A measure of how well-conditioned the problem is.
4.  **Solver Execution**:
    *   **Hybrid**: Uses the predicted rank $r$ (and neighbors $r \pm 1$) to attempt reconstruction.
    *   **Meta-Solver**: Generates multiple candidates (e.g., top-3 GNN predictions), reconstructs them, and uses a trained classifier to pick the best one based on residuals and GNN confidence.

## 📊 Recent Results

Recent experiments highlight the importance of **GNN-derived features** (Stability Score, Prediction Confidence) for the Meta-Solver. Removing these features leads to a significant drop in rank identification accuracy, especially in complex search spaces (e.g., "All" strategy), demonstrating that reconstruction residuals alone are insufficient for robust model selection in noisy environments.

---

# GNNESS: 图神经网络引导的稀疏求解器

本仓库包含了 **GNNESS** 的实现代码。这是一种利用图神经网络（GNN）来指导稀疏多项式插值和秩最小化问题经典求解器的新颖方法。通过预测含噪 Hankel 矩阵的潜在秩和稳定性，GNNESS 显著提高了参数重构的鲁棒性和效率。

## 🌟 核心特性

*   **RankGNN**：一种专门的 GNN 架构，用于处理 Hankel 图并预测潜在线性递推关系的秩。
*   **混合求解器 (Hybrid Solver)**：将 GNN 的秩预测与经典的基于 Sylvester 的求解器相结合，即使在存在噪声的情况下也能实现高精度求解。
*   **元求解器 (Meta-Solver)**：一个高级决策模块，使用元分类器根据 GNN 置信度、稳定性分数和重构残差，从多个候选解中（邻域、Top-3 或全部策略）选择最佳的重构结果。
*   **鲁棒性**：专为处理含噪数据而设计，能够解决经典方法（如 SVD 截断）经常失效的问题。

## 📂 目录结构

```
GNN-guided/
├── config.yaml                     # 模型和训练的主要配置文件
├── data_generation.py              # 合成数据生成（多项式、Hankel 矩阵）
├── graph_builder.py                # 从系数序列构建 Hankel 图
├── models.py                       # RankGNN 和基线模型的 PyTorch 实现
├── solver.py                       # 核心求解器逻辑（经典、SVD、混合、One-Shot）
├── train.py                        # RankGNN 模型训练脚本
├── evaluate.py                     # 所有方法的评估脚本，包括 Meta-Solver
├── run_paper_experiments.py        # 复现论文实验的脚本
├── run_no_gnn_features_experiment.py # 分析 GNN 特征对 Meta-Solver 影响的实验
├── utils.py                        # 工具函数
├── models/                         # 保存/加载训练好的模型权重的目录
│   ├── rank_gnn_fast.pth
│   └── ...
└── train_stats.pt                  # 用于特征归一化的已保存训练统计数据
```

## 🚀 快速开始

### 环境要求

确保已安装 Python 3.8+。本项目依赖以下关键库：

*   **PyTorch** (核心深度学习框架)
*   **PyTorch Geometric** (图神经网络操作)
*   **NumPy & SciPy** (数值线性代数)
*   **scikit-learn** (指标计算和元分类器)
*   **pandas** (结果数据处理)
*   **tqdm** (进度条)
*   **PyYAML** (配置管理)

通过 pip 安装依赖：

```bash
pip install torch torch-geometric numpy scipy scikit-learn pandas tqdm pyyaml matplotlib
```

*(注意：请根据您的 CUDA 版本/操作系统，从官方网站安装 PyTorch 和 PyTorch Geometric)*

### 🔧 配置

所有超参数均在 `config.yaml` 中管理。关键部分包括：

*   **`data`**：采样参数（次数范围 `d_range`，秩范围 `r_range`，噪声水平）。
*   **`model`**：GNN 架构设置（`hidden_dim`, `layers`, `dropout`）。
*   **`solver`**：求解器阈值（用于残差检查的 `tau`）。
*   **`training`**：学习率、批次大小、轮数。

## 🏃 使用指南

### 1. 训练模型

从头开始训练 RankGNN 模型：

```bash
python train.py
```

这将实时生成合成数据，训练模型，并将最佳权重保存到 `models/rank_gnn_fast.pth`（或配置的路径）。

### 2. 评估方法

评估经典方法、SVD、混合求解器和元求解器的性能：

```bash
python evaluate.py
```

该脚本将计算不同多项式次数下的准确率、有效解率 (VSR) 和运行时间。

### 3. 运行论文实验

复现论文中讨论的表格和图表（秩准确率、运行时间、噪声鲁棒性）：

```bash
python run_paper_experiments.py
```

### 4. Meta-Solver 特征分析

分析 GNN 特定特征（稳定性、置信度）对 Meta-Solver 性能的贡献：

```bash
python run_no_gnn_features_experiment.py
```

## 🧠 方法概述

1.  **输入**：含噪系数序列 $a = (a_0, a_1, \dots, a_n)$。
2.  **图构建**：将序列转换为 **Hankel 图**，其中节点代表系数，边代表 Hankel 矩阵结构。
3.  **秩预测**：**RankGNN** 处理图并预测：
    *   **秩 ($r$)**：指数项的数量。
    *   **稳定性分数**：衡量问题良态程度的指标。
4.  **求解器执行**：
    *   **混合求解器**：使用预测的秩 $r$（以及邻居 $r \pm 1$）尝试重构。
    *   **元求解器**：生成多个候选（例如 Top-3 GNN 预测），重构它们，并使用训练好的分类器根据残差和 GNN 置信度选择最佳结果。

## 📊 最新结果

最近的实验强调了 **GNN 衍生特征**（稳定性分数、预测置信度）对 Meta-Solver 的重要性。移除这些特征会导致秩识别准确率显著下降，尤其是在复杂的搜索空间（例如 "All" 策略）中，这表明在含噪环境中，仅靠重构残差不足以进行稳健的模型选择。
