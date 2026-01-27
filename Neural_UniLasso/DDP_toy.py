import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pyplot as plt

# ==========================================
# 1. 制造“玩具”数据 (Toy Data Generation)
# ==========================================
def get_toy_data(n_samples=500, n_features=10, valid_features=[0, 1, 2], noise_std=0.1):
    """
    制造一个回归数据集。
    只有 valid_features 里的变量是有用的，其他的都是噪音。
    关系是非线性的：y = 2*sin(x0) + 3*x1^2 - 1.5*x2 + noise
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # 构造真实标签
    # 变量0: 正弦关系
    # 变量1: 二次关系
    # 变量2: 线性关系
    # 变量3-9: 纯噪音
    y = (2.0 * np.sin(X[:, 0]) + 
         3.0 * (X[:, 1] ** 2) - 
         1.5 * X[:, 2] + 
         np.random.normal(0, noise_std, n_samples)).astype(np.float32)
    
    return torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1)

# ==========================================
# 2. 定义微分优化神经网络 (The Model)
# ==========================================
class DifferentiableLassoSelector(nn.Module):
    def __init__(self, num_features, hidden_dim=32, alpha=0.1):
        super().__init__()
        self.num_features = num_features
        self.alpha = alpha
        
        # 定义特征提取器 f_i (每个变量一个独立的小网络)
        self.feature_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),  # Tanh 比较适合这种简单的非线性拟合
                nn.Linear(hidden_dim, 1) 
            ) for _ in range(num_features)
        ])
        
        # === 核心修改：构建符合 DPP 规范的优化层 ===
        self.lasso_layer = self._build_dpp_layer(num_features)

    def _build_dpp_layer(self, n):
        lambda_var = cp.Variable(n)
        
        # 【修改点1】这里定义 L 参数 (Cholesky因子)，而不是 Q
        # 只要涉及参数相乘，必须非常小心。sum_squares(Affine) 是最稳的写法。
        L_param = cp.Parameter((n, n)) 
        p_param = cp.Parameter(n)
        
        # 【修改点2】目标函数改写
        # 原理: lambda^T * Q * lambda = || L^T * lambda ||^2
        # cp.sum_squares 保证了凸性，检查员(cvxpy)不会报错
        objective = cp.Minimize(0.5 * cp.sum_squares(L_param.T @ lambda_var) + p_param.T @ lambda_var)
        
        constraints = [lambda_var >= 0]
        
        problem = cp.Problem(objective, constraints)
        # 注意：一定要把 problem 声明清楚再传进去
        assert problem.is_dpp(), "这个定义如果不符合 DPP，就会在这里报错！"
        
        return CvxpyLayer(problem, parameters=[L_param, p_param], variables=[lambda_var])

    def forward(self, x, y):
        # x: (Batch, N), y: (Batch, 1)
        batch_size = x.shape[0]
        
        # Step 1: 提取特征 Z
        features = []
        for i in range(self.num_features):
            xi = x[:, i:i+1]
            fi = self.feature_nets[i](xi)
            features.append(fi)
        Z = torch.cat(features, dim=1) 
        
        # Step 2: 准备参数
        # 计算 Q = Z^T * Z
        Q = torch.matmul(Z.t(), Z)
        
        # 【关键】加一点点抖动 (Jitter)，防止矩阵奇异导致 Cholesky 失败
        # 在数值计算中，这是保证程序不崩的常用技巧
        jitter = 1e-4 * torch.eye(self.num_features, device=x.device)
        Q_stable = Q + jitter
        
        # 【修改点3】手动计算 Cholesky 分解: Q = L * L^T
        # L 是下三角矩阵
        try:
            L = torch.linalg.cholesky(Q_stable)
        except RuntimeError:
            # 万一还是失败了（极少情况），回退到特征值分解或者加大 jitter
            # 这里为了简单，我们直接加大抖动重试
            L = torch.linalg.cholesky(Q + 1e-2 * torch.eye(self.num_features, device=x.device))
        
        # 计算 p = alpha * 1 - Z^T * y
        # 注意根据 batch size 缩放 alpha，保持梯度量级一致
        scaled_alpha = self.alpha * batch_size 
        p = scaled_alpha * torch.ones(self.num_features, device=x.device) - torch.matmul(Z.t(), y).squeeze()
        
        # Step 3: 调用优化层 (传入 L 和 p)
        # 这一步 cvxpylayers 会自动处理反向传播
        lambda_star = self.lasso_layer(L, p)[0]
        
        # Step 4: 预测
        y_hat = torch.matmul(Z, lambda_star)
        
        return y_hat, lambda_star

# ==========================================
# 3. 训练与可视化 (Training & Visualization)
# ==========================================
def run_toy_experiment():
    # === 实验配置 ===
    NUM_FEATURES = 10
    VALID_IDX = [0, 1, 2] # 真实有效的变量索引
    LR = 0.01
    EPOCHS = 60           # 跑60轮差不多就能看出来了
    BATCH_SIZE = 100
    ALPHA = 0.5           # Lasso 惩罚力度，越大稀疏性越强
    
    # === 准备数据 ===
    X, y = get_toy_data(n_samples=500, n_features=NUM_FEATURES, valid_features=VALID_IDX)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # === 初始化模型 ===
    model = DifferentiableLassoSelector(num_features=NUM_FEATURES, alpha=ALPHA)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    # 记录数据用于画图
    history_lambda = [] 
    history_loss = []
    
    print(f"🚀 开始训练! 真实有效变量是: {VALID_IDX}")
    print(f"🔥 目标: 看着红线(有效变量)升起，灰线(噪音)归零...")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_lambdas = []
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward (自动解优化问题)
            y_pred, lambda_star = model(batch_X, batch_y)
            
            # Loss
            loss = loss_fn(y_pred.unsqueeze(1), batch_y)
            
            # Backward (梯度穿过优化层更新网络)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_lambdas.append(lambda_star.detach().numpy())
            
        # 记录统计
        avg_lambda = np.mean(epoch_lambdas, axis=0)
        history_lambda.append(avg_lambda)
        history_loss.append(epoch_loss / len(dataloader))
        
        if (epoch+1) % 10 == 0:
            top_indices = np.argsort(-avg_lambda)[:5]
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}")
            print(f"   -> Top Weights: 索引{top_indices} (值: {avg_lambda[top_indices].round(2)})")

    # ==========================================
    # 4. 绘图 (Visualization)
    # ==========================================
    history_lambda = np.array(history_lambda)
    
    plt.figure(figsize=(14, 6))
    
    # 左图: Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_loss, label='MSE Loss', color='black', linewidth=2)
    plt.title("Training Loss", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    # 右图: 变量权重演变
    plt.subplot(1, 2, 2)
    for i in range(NUM_FEATURES):
        if i in VALID_IDX:
            label = f"Feature {i} (Valid)"
            color = 'tab:red'
            alpha = 1.0
            linewidth = 3.0
            linestyle = '-'
        else:
            label = f"Feature {i} (Noise)" if i == 3 else None # 只标一个label避免图例太乱
            color = 'gray'
            alpha = 0.3
            linewidth = 1.0
            linestyle = '--'
        
        plt.plot(history_lambda[:, i], label=label, 
                 color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        
    plt.title("Evolution of Feature Weights (Lambda)", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Lasso Weight")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ 实验成功完成！")

if __name__ == "__main__":
    run_toy_experiment()