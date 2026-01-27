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
from copy import deepcopy

# 全局设置
torch.manual_seed(42)
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Arial'] # 防止绘图中文乱码(如果环境支持)
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 复杂数据生成 (高相关 + 交互 + 非线性)
# ==========================================
def generate_data(n_samples=1000, n_features=50, rho=0.8):
    # 生成 AR(1) 相关矩阵
    indices = np.arange(n_features)
    cov_matrix = rho ** np.abs(indices[:, None] - indices[None, :])
    X = np.random.multivariate_normal(np.zeros(n_features), cov_matrix, size=n_samples)
    
    # 构造真实信号
    # True Support: [0, 1, 5, 10, 11]
    f0 = -2.0 * X[:, 0]                     # Linear
    f1 = 2.0 * (X[:, 1]**2 - 1)            # Quadratic (Non-linear)
    f5 = 2.0 * np.sin(2 * X[:, 5])         # Sinusoidal
    f_inter = 3.0 * (X[:, 10] * X[:, 11])  # Interaction (乘法交互)
    
    # 加上较强的噪声，测试鲁棒性
    y = f0 + f1 + f5 + f_inter + np.random.normal(0, 1.5, n_samples)
    
    return X, y, [0, 1, 5, 10, 11]

# ==========================================
# 2. Neural UniLasso 模型定义
# ==========================================
class UnivariateNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 这里的子网络稍微简单一点，防止过拟合噪声
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.net(x)

class NeuralUniLasso(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.uni_nets = nn.ModuleList([UnivariateNetwork() for _ in range(n_features)])
        # 初始化非常小的权重，让它们从0开始竞争
        self.theta = nn.Parameter(torch.rand(n_features) * 0.001)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        z_list = []
        for i in range(self.n_features):
            z_list.append(self.uni_nets[i](x[:, i].view(-1, 1)))
        Z = torch.cat(z_list, dim=1)
        
        # 核心：Softplus 保证非负
        self.non_negative_theta = F.softplus(self.theta)
        y_pred = torch.matmul(Z, self.non_negative_theta) + self.bias
        return y_pred, self.non_negative_theta

# ==========================================
# 3. 训练流程 (带正则化路径)
# ==========================================
def train_path(X_train, y_train, X_val, y_val):
    n_features = X_train.shape[1]
    model = NeuralUniLasso(n_features)
    optimizer = optim.Adam(model.parameters(), lr=0.02) # 稍微加大 LR 加速收敛
    loss_fn = nn.MSELoss()
    
    X_t, y_t = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_v, y_v = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    
    # 策略调整：Lambda 范围整体上移，加大惩罚力度
    # 从 5.0 降到 0.05
    lambdas = np.logspace(0.7, -1.3, 20) 
    
    best_score = float('inf') # 这里的 score 综合考虑 MSE 和 稀疏度
    best_state = None
    best_lambda = 0
    
    path_hist = []
    
    print(f"训练开始 (搜索 {len(lambdas)} 个 Lambda)...")
    
    for l1_lambda in lambdas:
        # 每个 Lambda 训练 200 轮
        for _ in range(200):
            optimizer.zero_grad()
            y_pred, theta = model(X_t)
            mse = loss_fn(y_pred.view(-1), y_t)
            l1 = l1_lambda * torch.sum(theta)
            loss = mse + l1
            loss.backward()
            optimizer.step()
            
        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_pred, val_theta = model(X_v)
            val_mse = loss_fn(val_pred.view(-1), y_v).item()
            w = val_theta.numpy()
        model.train()
        
        # 记录
        sparsity = np.sum(w > 1e-3) # 仅用于记录，不用于选择
        path_hist.append({'lambda': l1_lambda, 'theta': w.copy(), 'mse': val_mse})
        
        # 模型选择策略：
        # 我们不仅看 MSE，还要看它是否够稀疏 (Occam's Razor)
        # Score = MSE * (1 + 0.05 * Active_Features) -> 稍微惩罚特征过多的模型
        # 或者简单点：直接选 Val MSE 最小的
        score = val_mse 
        
        if score < best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_lambda = l1_lambda
            
        if sparsity > n_features * 0.8: # 如果几乎全选了，说明 Lambda 太小了，没必要继续算了
            break
            
    model.load_state_dict(best_state)
    return model, path_hist, best_lambda

# ==========================================
# 4. 主实验逻辑 (含动态阈值筛选)
# ==========================================
def run_experiment():
    # A. 数据准备
    n_features = 50
    X, y, true_idx = generate_data(n_samples=800, n_features=n_features)
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # B. 训练 Neural UniLasso
    model, path_hist, best_lambda = train_path(X_train, y_train, X_val, y_val)
    
    # C. 训练 Lasso (对比)
    lasso = LassoCV(cv=5).fit(X_train, y_train)
    
    # D. 最终推断
    model.eval()
    with torch.no_grad():
        pred_neural, final_theta = model(torch.FloatTensor(X_test))
        w_neural = final_theta.numpy().flatten()
    
    mse_neural = mean_squared_error(y_test, pred_neural.numpy())
    mse_lasso = mean_squared_error(y_test, lasso.predict(X_test))
    
    # ==========================================
    # 核心修改：动态相对阈值 (Dynamic Relative Thresholding)
    # ==========================================
    
    # 策略：取最大权重的 10% 作为门槛
    # 这能有效切除深度学习中那些 0.001 级别的噪声底座
    relative_ratio = 0.4
    max_w = w_neural.max()
    dynamic_threshold = max_w * relative_ratio
    
    # Neural Selection
    sel_neural = np.where(w_neural > dynamic_threshold)[0]
    
    # Lasso Selection (Lasso 通常可以直接用非0，或者很小的阈值)
    sel_lasso = np.where(np.abs(lasso.coef_) > 1e-4)[0]
    
    # F1 计算
    def calc_f1(selected, true_set):
        y_true, y_pred = np.zeros(n_features), np.zeros(n_features)
        y_true[true_set] = 1
        y_pred[selected] = 1
        return f1_score(y_true, y_pred)
        
    f1_neural = calc_f1(sel_neural, true_idx)
    f1_lasso = calc_f1(sel_lasso, true_idx)
    
    # ==========================================
    # 5. 结果打印与可视化
    # ==========================================
    print("\n" + "="*50)
    print("FINAL RESULTS (With Dynamic Thresholding)")
    print("="*50)
    print(f"Dynamic Threshold: {dynamic_threshold:.4f} (10% of Max {max_w:.4f})")
    print(f"True Features:   {true_idx}")
    
    print(f"\n[Neural UniLasso]")
    print(f"MSE: {mse_neural:.4f}")
    print(f"Selected: {sel_neural}")
    print(f"Precision: {len(set(sel_neural) & set(true_idx))}/{len(sel_neural)}")
    print(f"F1 Score: {f1_neural:.4f}")
    
    print(f"\n[Standard Lasso]")
    print(f"MSE: {mse_lasso:.4f}")
    print(f"Selected: {sel_lasso}")
    print(f"F1 Score: {f1_lasso:.4f}")
    
    # --- 画图 ---
    plt.figure(figsize=(15, 6))
    
    # 左图：权重对比
    plt.subplot(1, 2, 1)
    # 归一化方便对比
    plt.bar(range(n_features), w_neural / max_w, color='blue', alpha=0.6, label='Neural UniLasso')
    plt.bar(range(n_features), -np.abs(lasso.coef_) / np.max(np.abs(lasso.coef_)), color='red', alpha=0.6, label='Lasso')
    
    # 画出阈值线
    plt.axhline(relative_ratio, color='green', linestyle='--', linewidth=2, label=f'Threshold ({relative_ratio*100}%)')
    plt.axhline(-0.01, color='green', linestyle='--', linewidth=2) # Lasso 阈值示意
    
    plt.title("Feature Importance & Selection Threshold")
    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Weight")
    plt.legend()
    
    # 右图：Elbow Plot (直观展示为何选这个阈值)
    plt.subplot(1, 2, 2)
    sorted_w = np.sort(w_neural)[::-1]
    plt.plot(sorted_w, 'o-', linewidth=2)
    plt.axhline(dynamic_threshold, color='green', linestyle='--', label='Cutoff Line')
    plt.title("Weight Magnitude (Sorted) - The Elbow")
    plt.xlabel("Rank")
    plt.ylabel("Absolute Weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()