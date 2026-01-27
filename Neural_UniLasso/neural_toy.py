import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 数据生成：模拟复杂的非线性关系
# ==========================================
def generate_nonlinear_data(n_samples=500, n_features=10):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # 真实关系构造：
    # 1. 特征 0: 线性正相关 (UniLasso 擅长)
    # 2. 特征 1: 二次函数关系 (x^2)，原版 UniLasso 会失效，因为线性相关性低
    # 3. 特征 2: 阶跃/阈值关系 (Sigmoid)，模拟生物饱和效应
    # 4. 其他特征: 纯噪声
    
    y = (3.0 * X[:, 0] + 
         2.0 * (X[:, 1]**2) + 
         4.0 * np.tanh(X[:, 2]) + 
         np.random.normal(0, 0.1, n_samples))
    
    return X, y

# ==========================================
# 2. 定义 Neural UniLasso 模型架构
# ==========================================
class UnivariateSubNet(nn.Module):
    """处理单个特征的微型神经网络"""
    def __init__(self):
        super().__init__()
        # 简单的 MLP: 1 -> 8 -> 1
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1) # 输出变换后的单变量特征 z_j
        )
    
    def forward(self, x):
        return self.net(x)

class NeuralUniLasso(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        
        # 第一步：并行单变量变换层
        # ModuleList 允许我们将每个特征的网络独立存储
        self.univariate_nets = nn.ModuleList([
            UnivariateSubNet() for _ in range(n_features)
        ])
        
        # 第二步：非负融合层 (Fusion Layer)
        # 初始化一个形状为 (n_features, ) 的权重向量
        self.fusion_weights = nn.Parameter(torch.rand(n_features))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 1. 独立处理每个特征
        transformed_features = []
        for i in range(self.n_features):
            # 取出第 i 列，保持维度 (Batch, 1)
            feat_i = x[:, i].unsqueeze(1) 
            # 通过对应的子网络
            z_i = self.univariate_nets[i](feat_i) 
            transformed_features.append(z_i)
        
        # 拼接回矩阵 Z: (Batch, n_features)
        Z = torch.cat(transformed_features, dim=1)
        
        # 2. 融合 (Fusion)
        # 强制权重非负 (模拟 Non-negative Constraint)
        # 使用 softplus 或 relu 保证 theta >= 0
        positive_weights = torch.relu(self.fusion_weights)
        
        # 线性组合: y_pred = Z * theta + b
        # 这里的 positive_weights 就是我们最终的稀疏系数
        y_pred = torch.matmul(Z, positive_weights) + self.bias
        
        return y_pred, Z, positive_weights

# ==========================================
# 3. 训练过程 (端到端优化)
# ==========================================
def train_experiment():
    # 准备数据
    X_raw, y_raw = generate_nonlinear_data()
    
    # 转换为 Tensor
    X_tensor = torch.FloatTensor(X_raw)
    y_tensor = torch.FloatTensor(y_raw)
    
    model = NeuralUniLasso(n_features=X_raw.shape[1])
    
    # 优化器：同时优化 单变量网络参数 和 融合层权重
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 超参数
    lambda_l1 = 0.1  # Lasso 稀疏惩罚系数
    epochs = 1000
    
    loss_history = []
    
    print(f"{'Epoch':<10} | {'Total Loss':<12} | {'MSE Loss':<12} | {'L1 Penalty':<12}")
    print("-" * 55)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        y_pred, Z_transformed, current_weights = model(X_tensor)
        
        # 1. 预测误差 (MSE)
        mse_loss = nn.MSELoss()(y_pred, y_tensor)
        
        # 2. 稀疏惩罚 (L1 Regularization on Fusion Weights)
        # 只对融合层的权重施加 L1，模拟 Lasso
        l1_loss = torch.sum(torch.abs(current_weights))
        
        # 总 Loss
        total_loss = mse_loss + lambda_l1 * l1_loss
        
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if epoch % 100 == 0:
            print(f"{epoch:<10} | {total_loss.item():<12.4f} | {mse_loss.item():<12.4f} | {l1_loss.item():<12.4f}")

    return model, X_tensor, y_tensor, loss_history

# ==========================================
# 4. 可视化分析 (Dashboard)
# ==========================================
def visualize_results(model, X, y, loss_hist):
    model.eval()
    with torch.no_grad():
        y_pred, Z, final_weights = model(X)
        final_weights = final_weights.numpy()
        Z = Z.numpy()
        X = X.numpy()
    
    plt.figure(figsize=(15, 10))
    plt.suptitle("Neural UniLasso Experiment Results", fontsize=16)

    # 图1: 训练 Loss 曲线
    plt.subplot(2, 2, 1)
    plt.plot(loss_hist, label='Total Loss', color='blue')
    plt.title("Training Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    # 图2: 最终稀疏权重 (Fusion Weights)
    plt.subplot(2, 2, 2)
    colors = ['green' if i < 3 else 'gray' for i in range(len(final_weights))]
    bars = plt.bar(range(len(final_weights)), final_weights, color=colors)
    plt.title("Learned Fusion Weights (Sparsity Check)")
    plt.xlabel("Feature Index")
    plt.ylabel("Weight Magnitude")
    plt.xticks(range(len(final_weights)))
    # 标注真实有用的特征
    plt.text(0, final_weights[0], "Linear", ha='center', va='bottom')
    plt.text(1, final_weights[1], "Quadratic", ha='center', va='bottom')
    plt.text(2, final_weights[2], "Tanh", ha='center', va='bottom')

    # 图3 & 4: 关键特征的“学习到的变换”
    # 我们画出 原始特征 X vs 变换后特征 Z
    # 如果 Neural UniLasso 工作正常，它应该把非线性关系“拉直”
    
    # 特征 1 (Quadratic x^2)
    plt.subplot(2, 2, 3)
    plt.scatter(X[:, 1], Z[:, 1], alpha=0.5, color='orange')
    plt.title(f"Feature 1 Transformation\n(Original Quadratic -> Learned Feature)")
    plt.xlabel("Original x1 (Input)")
    plt.ylabel("Transformed z1 (Output of SubNet)")
    plt.grid(True)

    # 特征 2 (Tanh)
    plt.subplot(2, 2, 4)
    plt.scatter(X[:, 2], Z[:, 2], alpha=0.5, color='purple')
    plt.title(f"Feature 2 Transformation\n(Original Tanh -> Learned Feature)")
    plt.xlabel("Original x2 (Input)")
    plt.ylabel("Transformed z2 (Output of SubNet)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n====== 最终参数输出 ======")
    print("Feature Index | Learned Weight | True Importance")
    print("-" * 45)
    print(f"0 (Linear)    | {final_weights[0]:.4f}         | High (Linear)")
    print(f"1 (Quadratic) | {final_weights[1]:.4f}         | High (Non-linear)")
    print(f"2 (Tanh)      | {final_weights[2]:.4f}         | High (Non-linear)")
    for i in range(3, len(final_weights)):
        print(f"{i} (Noise)     | {final_weights[i]:.4f}         | Zero")

# 运行实验
if __name__ == "__main__":
    trained_model, X_dat, y_dat, history = train_experiment()
    visualize_results(trained_model, X_dat, y_dat, history)