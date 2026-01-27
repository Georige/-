import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, f1_score

# 全局设置
torch.manual_seed(42)
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. AR(1) 高相关数据生成器
# ==========================================
def generate_ar1_data(n_samples=1000, n_features=50, rho=0.9):
    """
    生成 AR(1) 相关结构的数据：Cov(i,j) = rho^|i-j|
    rho=0.9 意味着相邻特征极其相似，特征选择难度极大。
    """
    # 1. 构造协方差矩阵
    indices = np.arange(n_features)
    cov_matrix = rho ** np.abs(indices[:, None] - indices[None, :])
    
    # 2. 生成特征 X
    X = np.random.multivariate_normal(np.zeros(n_features), cov_matrix, size=n_samples)
    
    # 3. 构造混合关系的 Y
    # 真实支持集 Support: [5, 6, 15, 20]
    # 注意：5和6高度相关 (rho=0.9)，测试模型能否区分不同形状
    
    # Feature 5: Linear (线性)
    f5 = 3.0 * X[:, 5]
    
    # Feature 6: Quadratic (非线性 U型)
    # 挑战：X5 和 X6 高度相关。Lasso 可能会因为共线性随机选一个。
    # 但 Neural UniLasso 应该能发现 X6 有独特的 U 型贡献，从而同时保留两者，或者正确区分。
    f6 = 2.0 * (X[:, 6]**2 - 1)
    
    # Feature 15: Sinusoidal (周期性)
    f15 = 4.0 * np.sin(2.0 * X[:, 15])
    
    # Feature 20: Tanh (饱和)
    f20 = 3.0 * np.tanh(2.0 * X[:, 20])
    
    # 合成 Y
    y = f5 + f6 + f15 + f20 + np.random.normal(0, 1.0, n_samples)
    
    true_features = [5, 6, 15, 20]
    return X, y, true_features, cov_matrix

# ==========================================
# 2. Neural UniLasso 模型 (保持架构一致)
# ==========================================
class UnivariateNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x)

class NeuralUniLasso(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.uni_nets = nn.ModuleList([UnivariateNetwork() for _ in range(n_features)])
        self.theta = nn.Parameter(torch.rand(n_features) * 0.01) # Small init
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        z_list = [self.uni_nets[i](x[:, i].view(-1, 1)) for i in range(self.n_features)]
        Z = torch.cat(z_list, dim=1)
        self.weights = F.softplus(self.theta) # Constraint
        return torch.matmul(Z, self.weights) + self.bias, self.weights, Z

# ==========================================
# 3. 训练与可视化流程
# ==========================================
def run_ar1_experiment():
    # A. 准备数据
    rho = 0.85 # 设置极高相关性
    X, y, true_idx, cov_mat = generate_ar1_data(n_samples=1000, n_features=50, rho=rho)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # B. 训练 Neural UniLasso
    model = NeuralUniLasso(50)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Neural UniLasso on AR(1) Data...")
    # 简单的训练循环 (为了演示省略了 CV Path, 实际建议使用)
    for epoch in range(1500):
        optimizer.zero_grad()
        y_pred, w, _ = model(torch.FloatTensor(X_train))
        mse = nn.MSELoss()(y_pred.view(-1), torch.FloatTensor(y_train))
        l1 = 0.1 * torch.sum(w) # Lambda=0.1
        loss = mse + l1
        loss.backward()
        optimizer.step()
        
    # C. 训练 Lasso 对比
    lasso = LassoCV(cv=5).fit(X_train, y_train)
    
    # D. 结果分析
    model.eval()
    with torch.no_grad():
        _, w_tensor, _ = model(torch.FloatTensor(X_test))
        w_neural = w_tensor.numpy()
        
    # 动态阈值筛选
    threshold = w_neural.max() * 0.1
    sel_neural = np.where(w_neural > threshold)[0]
    sel_lasso = np.where(np.abs(lasso.coef_) > 1e-4)[0]
    
    print(f"\nTrue Features: {true_idx}")
    print(f"Neural Selected: {sel_neural}")
    print(f"Lasso Selected:  {sel_lasso}")
    
    # ==========================================
    # 可视化仪表盘
    # ==========================================
    plt.figure(figsize=(16, 10))
    plt.suptitle(f"AR(1) Correlation Experiment (rho={rho})", fontsize=16)
    
    # 1. 相关性热图 (证明数据很难)
    plt.subplot(2, 3, 1)
    sns.heatmap(cov_mat[:10, :10], cmap="coolwarm", annot=True, fmt=".1f")
    plt.title("Correlation Matrix (First 10 feats)")
    
    # 2. 特征权重对比
    plt.subplot(2, 3, 2)
    plt.bar(range(50), w_neural/w_neural.max(), color='blue', alpha=0.6, label='Neural')
    plt.bar(range(50), -np.abs(lasso.coef_)/np.max(np.abs(lasso.coef_)), color='red', alpha=0.6, label='Lasso')
    plt.axhline(0, color='k')
    for i in true_idx: plt.axvline(i, color='green', linestyle='--', alpha=0.3)
    plt.legend()
    plt.title("Feature Selection (Neural vs Lasso)")
    
    # 3. 关键验证：Feature 5 (Linear) vs Feature 6 (Quadratic)
    # 这两个特征高度相关，看看模型学到了什么形状
    x_grid = torch.linspace(-3, 3, 100).view(-1, 1)
    
    plt.subplot(2, 3, 4)
    with torch.no_grad(): z5 = model.uni_nets[5](x_grid).numpy()
    plt.plot(x_grid, z5, 'b-', lw=3)
    plt.title("Learned Shape: Feat 5 (True: Linear)")
    plt.xlabel("Input x")
    
    plt.subplot(2, 3, 5)
    with torch.no_grad(): z6 = model.uni_nets[6](x_grid).numpy()
    plt.plot(x_grid, z6, 'r-', lw=3)
    plt.title("Learned Shape: Feat 6 (True: Quadratic)")
    plt.xlabel("Input x")
    
    # 4. 看看邻居 Feature 4 (噪声)
    # 它和 Feature 5 的相关性也高达 rho，看看是否被抑制
    plt.subplot(2, 3, 6)
    with torch.no_grad(): z4 = model.uni_nets[4](x_grid).numpy()
    plt.plot(x_grid, z4, 'gray', lw=3)
    plt.title(f"Learned Shape: Feat 4 (Noise, Corr={rho} w/ F5)")
    plt.ylim(plt.ylim()) # 保持比例
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_ar1_experiment()