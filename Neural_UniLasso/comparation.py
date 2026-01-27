import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 数据生成 (保持不变)
# ==========================================
def generate_complex_data(n_samples=1000, n_features=50):
    X = np.random.uniform(-3, 3, (n_samples, n_features))
    
    # Feature 0: Linear Positive
    f0 = 3.0 * X[:, 0]
    
    # Feature 1: Quadratic (U-shape)
    f1 = 2.0 * (X[:, 1] ** 2) - 4.0 
    
    # Feature 5: Linear Negative (关键测试点)
    # 如果有非负约束，子网络必须学会翻转。
    # 如果无约束，子网络可能输出正，权重变负。
    f5 = -2.0 * X[:, 5] 
    
    # 其他非线性
    f2 = 3.0 * np.sin(1.5 * X[:, 2])
    f3 = 4.0 * np.tanh(2.0 * X[:, 3])
    f4 = -2.5 * np.abs(X[:, 4]) + 2.0
    
    y = f0 + f1 + f2 + f3 + f4 + f5 + np.random.normal(0, 0.5, n_samples)
    
    # 真实支持集
    true_support = [0, 1, 2, 3, 4, 5]
    return X, y, true_support

# ==========================================
# 2. 通用模型定义 (支持开关约束)
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

class NeuralGAM(nn.Module):
    def __init__(self, n_features, non_negative=True):
        super().__init__()
        self.n_features = n_features
        self.non_negative = non_negative # 开关
        
        self.uni_nets = nn.ModuleList([UnivariateNetwork() for _ in range(n_features)])
        
        # 初始化
        if self.non_negative:
            # 如果非负，初始化为小的正数
            self.theta = nn.Parameter(torch.rand(n_features) * 0.1)
        else:
            # 如果无约束，初始化为均值为0的分布，允许正负
            self.theta = nn.Parameter(torch.randn(n_features) * 0.01)
            
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        z_list = []
        for i in range(self.n_features):
            z_list.append(self.uni_nets[i](x[:, i].view(-1, 1)))
        Z = torch.cat(z_list, dim=1)
        
        if self.non_negative:
            # 有约束：使用 Softplus
            self.effective_theta = F.softplus(self.theta)
        else:
            # 无约束：直接使用 Theta
            self.effective_theta = self.theta
            
        y_pred = torch.matmul(Z, self.effective_theta) + self.bias
        return y_pred, self.effective_theta

# ==========================================
# 3. 训练函数 (适配两种模式)
# ==========================================
def train_model(X, y, non_negative=True, epochs=3000, lr=0.005, l1_lambda=0.1):
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    
    model = NeuralGAM(X.shape[1], non_negative=non_negative)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    print(f"Training [{'With' if non_negative else 'Without'} Non-negative Constraint]...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred, theta = model(X_t)
        
        mse = loss_fn(y_pred.view(-1), y_t)
        
        # 核心区别：L1 惩罚的计算
        if non_negative:
            l1_reg = l1_lambda * torch.sum(theta) # theta 恒正
        else:
            l1_reg = l1_lambda * torch.sum(torch.abs(theta)) # theta 有正负，需取绝对值
            
        loss = mse + l1_reg
        loss.backward()
        optimizer.step()
        
    return model

# ==========================================
# 4. 对比实验
# ==========================================
def run_comparison():
    # 数据
    X, y, true_idx = generate_complex_data(n_samples=1000, n_features=50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 训练两个模型
    model_constrained = train_model(X_train, y_train, non_negative=True, l1_lambda=0.05)
    model_unconstrained = train_model(X_train, y_train, non_negative=False, l1_lambda=0.05)
    
    # 提取权重
    with torch.no_grad():
        _, theta_c = model_constrained(torch.FloatTensor(X_test))
        _, theta_u = model_unconstrained(torch.FloatTensor(X_test))
        
        w_c = theta_c.numpy()
        w_u = theta_u.numpy() # 注意：这里可能有负数
        
    # 计算 MSE
    mse_c = mean_squared_error(y_test, model_constrained(torch.FloatTensor(X_test))[0].detach().numpy())
    mse_u = mean_squared_error(y_test, model_unconstrained(torch.FloatTensor(X_test))[0].detach().numpy())

    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    print(f"MSE (With Constraint):    {mse_c:.4f}")
    print(f"MSE (Without Constraint): {mse_u:.4f}")
    
    # ==========================================
    # 5. 关键可视化
    # ==========================================
    plt.figure(figsize=(15, 10))
    plt.suptitle("Impact of Non-negative Constraint on Feature Weights", fontsize=16)
    
    # 图1：权重对比 (条形图)
    plt.subplot(2, 2, 1)
    plt.bar(range(50), w_c, color='blue', alpha=0.6, label='With Constraint (>=0)')
    # 对于无约束模型，我们画出它的原始值（可能有负数）
    plt.bar(range(50), w_u, color='orange', alpha=0.6, label='Without Constraint')
    plt.axhline(0, color='k', lw=1)
    plt.title("Learned Feature Weights (Theta)")
    plt.legend()
    plt.xlabel("Feature Index")
    
    # 图2：Feature 5 (Linear Negative) 的子网络形状对比
    # 真实关系是 y = -2 * x5
    # 有约束模型：theta > 0，所以子网络必须学会输出 "-x" 的形状
    # 无约束模型：theta 可以 < 0，子网络可能输出 "x"，然后靠 theta 翻转
    idx = 5
    x_range = torch.linspace(-3, 3, 100).view(-1, 1)
    
    with torch.no_grad():
        z_c = model_constrained.uni_nets[idx](x_range).numpy()
        z_u = model_unconstrained.uni_nets[idx](x_range).numpy()
    
    plt.subplot(2, 2, 2)
    plt.plot(x_range, -2 * x_range, 'k--', label='True Relationship (-2x)', linewidth=1)
    # 注意：最终贡献 = z * theta
    # 我们画出 "有效贡献" (Effective Contribution)
    eff_c = z_c * w_c[idx]
    eff_u = z_u * w_u[idx]
    
    plt.plot(x_range, eff_c, 'b-', label='Constrained Model (Total Effect)')
    plt.plot(x_range, eff_u, 'orange', linestyle='--', label='Unconstrained Model (Total Effect)')
    plt.title(f"Effective Contribution of Feature {idx} (Negative Linear)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 图3：Feature 5 的 "内部视角" (Raw Subnetwork Output)
    # 这是为了验证符号二义性
    plt.subplot(2, 2, 3)
    plt.plot(x_range, z_c, 'b-', label='SubNet Output (Constrained)')
    plt.plot(x_range, z_u, 'orange', label='SubNet Output (Unconstrained)')
    plt.title(f"Internal Subnetwork Output (Phi(x)) for Feat {idx}")
    plt.xlabel("Input x")
    plt.ylabel("Output z")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 图4：Feature 1 (Quadratic) 的对比
    idx = 1
    with torch.no_grad():
        z_c = model_constrained.uni_nets[idx](x_range).numpy()
        z_u = model_unconstrained.uni_nets[idx](x_range).numpy()
        
    plt.subplot(2, 2, 4)
    plt.plot(x_range, z_c, 'b-', label='SubNet Output (Constrained)')
    plt.plot(x_range, z_u, 'orange', label='SubNet Output (Unconstrained)')
    plt.title(f"Internal Subnetwork Output for Feat {idx} (Quadratic)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()