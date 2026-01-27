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

# 随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 高相关 & 交互项 数据生成器
# ==========================================
def generate_correlated_interaction_data(n_samples=1000, n_features=50, rho=0.8):
    """
    生成具有高相关性特征和交互项的数据
    """
    # A. 生成高相关特征矩阵 (AR(1) Covariance Structure)
    # Cov(i, j) = rho^|i-j|
    indices = np.arange(n_features)
    cov_matrix = rho ** np.abs(indices[:, None] - indices[None, :])
    
    # 生成 X
    X = np.random.multivariate_normal(np.zeros(n_features), cov_matrix, size=n_samples)
    
    # B. 构造复杂的 Y (加性 + 交互)
    # 真实支持集: [0, 1, 5, 10, 11]
    # 其中 10 和 11 是交互对
    
    # 1. Linear (on Feature 0)
    f0 = 2.0 * X[:, 0]
    
    # 2. Nonlinear: Quadratic (on Feature 1)
    f1 = 1.5 * (X[:, 1]**2 - 1)
    
    # 3. Nonlinear: Sinusoidal (on Feature 5)
    f5 = 2.0 * np.sin(2 * X[:, 5])
    
    # 4. Interaction Term (Feature 10 * Feature 11)
    # 这两个特征本身相关性就很高 (rho^1 = 0.8)，且以乘积形式影响 y
    # 加性模型通常很难处理这个，除非它能分别拟合出某种非线性主效应
    f_inter = 3.0 * (X[:, 10] * X[:, 11])
    
    # 合成 Y
    y = f0 + f1 + f5 + f_inter + np.random.normal(0, 1.0, n_samples)
    
    true_support = [0, 1, 5, 10, 11]
    
    return X, y, true_support, cov_matrix

# ==========================================
# 2. Neural UniLasso 模型 (复用之前的架构)
# ==========================================
class UnivariateNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ELU(),
            nn.Linear(16, 8),
            nn.ELU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x)

class NeuralUniLasso(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.uni_nets = nn.ModuleList([UnivariateNetwork() for _ in range(n_features)])
        self.theta = nn.Parameter(torch.rand(n_features) * 0.01) # 初始化小一点
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        z_list = []
        for i in range(self.n_features):
            feat_i = x[:, i].view(-1, 1)
            z_list.append(self.uni_nets[i](feat_i))
        Z = torch.cat(z_list, dim=1)
        
        # 强制非负
        self.non_negative_theta = F.softplus(self.theta)
        y_pred = torch.matmul(Z, self.non_negative_theta) + self.bias
        return y_pred, Z, self.non_negative_theta

# ==========================================
# 3. 训练函数
# ==========================================
def train_model(X, y, l1_lambda=0.05, epochs=1500, lr=0.01):
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    
    model = NeuralUniLasso(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    print(f"Training Start (L1={l1_lambda})...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred, _, theta = model(X_t)
        
        mse = loss_fn(y_pred.view(-1), y_t)
        l1 = l1_lambda * torch.sum(theta)
        loss = mse + l1
        
        loss.backward()
        optimizer.step()
        
    return model

# ==========================================
# 4. 主实验流程
# ==========================================
def run_stress_test():
    # 1. 生成数据
    n_features = 50
    X, y, true_idx, cov_mat = generate_correlated_interaction_data(n_samples=800, n_features=n_features)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 2. 训练 Neural UniLasso
    neural_model = train_model(X_train, y_train, l1_lambda=0.15) # 稍微加大惩罚以应对共线性
    
    # 3. 训练 Lasso (Baseline)
    lasso = LassoCV(cv=5).fit(X_train, y_train)
    
    # 4. 评估
    neural_model.eval()
    with torch.no_grad():
        pred_neural, _, theta = neural_model(torch.FloatTensor(X_test))
        w_neural = theta.numpy()
        
    mse_neural = mean_squared_error(y_test, pred_neural.numpy().flatten())
    mse_lasso = mean_squared_error(y_test, lasso.predict(X_test))
    
    # 特征筛选性能 (F1 Score)
    # 归一化权重以便比较
    w_neural_norm = w_neural / w_neural.max()
    w_lasso_norm = np.abs(lasso.coef_) / np.abs(lasso.coef_).max()
    
    # 设定选择阈值 (Top K or Value Threshold)
    threshold = 0.25
    sel_neural = np.where(w_neural_norm > threshold)[0]
    sel_lasso = np.where(w_lasso_norm > threshold)[0]
    
    # 计算 Precision/Recall/F1
    def get_metrics(selected, true_set, p):
        y_true = np.zeros(p)
        y_true[true_set] = 1
        y_pred = np.zeros(p)
        y_pred[selected] = 1
        return f1_score(y_true, y_pred)
    
    f1_neural = get_metrics(sel_neural, true_idx, n_features)
    f1_lasso = get_metrics(sel_lasso, true_idx, n_features)

    print("\n" + "="*50)
    print("STRESS TEST RESULTS: Correlation + Interaction + Nonlinearity")
    print("="*50)
    print(f"True Support: {true_idx} (Note: 10 & 11 are interaction terms)")
    print(f"\n[Neural UniLasso]")
    print(f"MSE: {mse_neural:.4f}")
    print(f"Selected Feats: {sel_neural}")
    print(f"F1 Score: {f1_neural:.4f}")
    
    print(f"\n[Standard Lasso]")
    print(f"MSE: {mse_lasso:.4f}")
    print(f"Selected Feats: {sel_lasso}")
    print(f"F1 Score: {f1_lasso:.4f}")
    
    # ==========================================
    # 5. 可视化分析
    # ==========================================
    plt.figure(figsize=(15, 10))
    plt.suptitle("Neural UniLasso Stress Test: High Correlation & Interactions", fontsize=16)
    
    # 1. 特征相关性热图 (证明数据很难)
    plt.subplot(2, 2, 1)
    sns.heatmap(cov_mat[:15, :15], cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap (First 15 feats)")
    
    # 2. 特征权重对比
    plt.subplot(2, 2, 2)
    plt.bar(range(n_features), w_neural_norm, color='blue', alpha=0.6, label='Neural UniLasso')
    plt.bar(range(n_features), -w_lasso_norm, color='red', alpha=0.6, label='Standard Lasso')
    plt.axhline(0, color='k')
    # 标记真实特征
    for i in true_idx:
        plt.axvline(i, color='green', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title("Feature Importance (Up=Neural, Down=Lasso)")
    plt.xlabel("Feature Index")
    
    # 3. 交互项分析 (Feature 10 vs 11)
    # 看看 Neural UniLasso 学到了什么形状来逼近乘法
    x_range = torch.linspace(-3, 3, 100).view(-1, 1)
    with torch.no_grad():
        z10 = neural_model.uni_nets[10](x_range).numpy()
        z11 = neural_model.uni_nets[11](x_range).numpy()
        
    plt.subplot(2, 2, 3)
    plt.plot(x_range.numpy(), z10, label="Learned Phi(x10)", color="orange")
    plt.plot(x_range.numpy(), z11, label="Learned Phi(x11)", color="purple")
    plt.title("Learned Transformations for Interaction Terms")
    plt.xlabel("Input Value")
    plt.ylabel("Transformed Output")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 散点图：真实值 vs 预测值
    plt.subplot(2, 2, 4)
    plt.scatter(y_test, lasso.predict(X_test), alpha=0.5, label='Lasso', color='red', s=10)
    plt.scatter(y_test, pred_neural.numpy().flatten(), alpha=0.5, label='Neural', color='blue', s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.title("Prediction vs Truth")
    plt.xlabel("True Y")
    plt.ylabel("Predicted Y")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_stress_test()

    