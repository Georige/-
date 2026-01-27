import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm  # 进度条
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, f1_score, recall_score, precision_score

# ==========================================
# 0. 实验配置表 (Experiment Configuration)
# 修改这里相当于修改整个实验
# ==========================================
EXPERIMENT_CONFIG = {
    # -- 模拟数据参数 --
    "n_train": 1000,           # 训练集大小
    "n_test": 2000,            # 测试集大小
    "p_features": 200,         # 特征总维度
    "data_range": (-3, 3),     # U(-3, 3) 均匀分布
    "noise_level": 0.5,        # 噪声标准差
    
    # -- 神经网络架构参数 --
    "hidden_layers": [16, 8],  # 隐藏层结构
    "activation": "ELU",       # 激活函数
    
    # -- 训练超参数 --
    "epochs": 2000,            # 迭代轮数
    "learning_rate": 0.01,     # 学习率
    "l1_lambda": 0.05,         # L1 正则化强度 (稀疏惩罚)
    
    # -- 变量选择 --
    "lasso_threshold": 1e-4,   # Lasso 的硬阈值
    "rf_estimators": 100,      # 随机森林树的数量
    
    # -- 其他 --
    "seed": 42                 # 随机种子
}

# 全局设置
torch.manual_seed(EXPERIMENT_CONFIG["seed"])
np.random.seed(EXPERIMENT_CONFIG["seed"])
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")

# ==========================================
# 1. 核心模型：Neural UniLasso
# ==========================================
class UnivariateNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 结构：1 -> 16 -> 8 -> 1 (ELU激活)
        h1, h2 = EXPERIMENT_CONFIG["hidden_layers"]
        self.net = nn.Sequential(
            nn.Linear(1, h1),
            nn.ELU(),
            nn.Linear(h1, h2),
            nn.ELU(),
            nn.Linear(h2, 1)
        )
    def forward(self, x):
        return self.net(x)

class NeuralUniLasso(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        # 并行子网络
        self.uni_nets = nn.ModuleList([UnivariateNetwork() for _ in range(n_features)])
        # 融合权重 (初始化为小正数)
        self.theta = nn.Parameter(torch.rand(n_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x: [batch, p]
        z_list = []
        for i in range(self.n_features):
            z_list.append(self.uni_nets[i](x[:, i].view(-1, 1)))
        
        Z = torch.cat(z_list, dim=1) 
        
        # 非负约束 (Softplus)
        self.weights = F.softplus(self.theta)
        
        y_pred = torch.matmul(Z, self.weights) + self.bias
        return y_pred, self.weights, Z

# ==========================================
# 2. 数据生成器
# ==========================================
def generate_data(config):
    n = config["n_train"] + config["n_test"]
    p = config["p_features"]
    low, high = config["data_range"]
    
    # X ~ U(-3, 3)
    X = np.random.uniform(low, high, size=(n, p))
    
    # 真实变量
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    
    # 真实方程: y = sin(x1) - 3*tan(x2) + 5*e^x3 - 2*x_4
    t1 = np.sin(x1)
    
    # tan(x) 在 +/- 1.57 处爆炸，必须截断
    t2 = -3 * np.clip(np.tan(x2), -10, 10) 
    
    t3 = 5 * np.exp(x3)
    t4 = -2 * x4
    
    y_raw = t1 + t2 + t3 + t4
    y = y_raw + np.random.normal(0, config["noise_level"], n)
    
    true_idx = [0, 1, 2, 3]
    
    return X[:config["n_train"]], X[config["n_train"]:], \
           y[:config["n_train"]], y[config["n_train"]:], \
           true_idx

# ==========================================
# 3. 辅助函数：最大 Gap 截断与指标
# ==========================================
def get_max_gap_threshold(weights):
    """最大 Gap 截断策略"""
    w_abs = np.abs(weights)
    # 降序排列
    sorted_w = np.sort(w_abs)[::-1]
    
    # 忽略非常小的权重 (比如小于最大权重的 1%)
    valid_mask = sorted_w > (sorted_w.max() * 0.01)
    if valid_mask.sum() < 2: return 0.0
    
    w_valid = sorted_w[valid_mask]
    
    # 计算相邻差值 (Gap)
    gaps = -np.diff(w_valid) # w[i] - w[i+1]
    best_gap_idx = np.argmax(gaps)
    
    # 阈值取 gap 处的中间值
    threshold = (w_valid[best_gap_idx] + w_valid[best_gap_idx+1]) / 2
    return threshold

def calculate_metrics(y_true, y_pred, selected_idx, true_idx, p_total):
    mse = mean_squared_error(y_true, y_pred)
    
    # 转换为 0/1 向量计算分类指标
    y_true_bin = np.zeros(p_total)
    y_true_bin[true_idx] = 1
    
    y_pred_bin = np.zeros(p_total)
    y_pred_bin[selected_idx] = 1
    
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    sensitivity = recall_score(y_true_bin, y_pred_bin, zero_division=0) # Recall
    
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {"MSE": mse, "F1": f1, "Sensitivity": sensitivity, "Specificity": specificity, "Selected_Num": len(selected_idx)}

# ==========================================
# 4. 实验主程序
# ==========================================
def run_experiment():
    cfg = EXPERIMENT_CONFIG
    print(f"Loading Configuration... N={cfg['n_train']}, P={cfg['p_features']}, Range={cfg['data_range']}")
    
    # 1. 数据生成
    X_train, X_test, y_train, y_test, true_idx = generate_data(cfg)
    
    # 标准化 (对 NN 训练至关重要)
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    # 为了计算真实 MSE，我们在预测后反变换，或者在这里保持测试集 y 的原始尺度
    
    # --- 模型 1: Neural UniLasso ---
    print("\n[1/4] Training Neural UniLasso...")
    nu_model = NeuralUniLasso(cfg["p_features"])
    opt = optim.Adam(nu_model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.MSELoss()
    
    Xt = torch.FloatTensor(X_train_s)
    yt = torch.FloatTensor(y_train_s)
    
    # 进度条
    pbar = tqdm(range(cfg["epochs"]), desc="Neural UniLasso Epochs")
    for epoch in pbar:
        opt.zero_grad()
        pred, w, _ = nu_model(Xt)
        mse = loss_fn(pred.view(-1), yt)
        reg = cfg["l1_lambda"] * torch.sum(w)
        loss = mse + reg
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Sparsity": f"{(w>0.01).sum().item()}"})
            
    # 推理
    nu_model.eval()
    with torch.no_grad():
        _, w_nu_tensor, _ = nu_model(torch.FloatTensor(X_test_s))
        w_nu = w_nu_tensor.numpy()
        pred_nu_scaled = nu_model(torch.FloatTensor(X_test_s))[0].numpy().flatten()
        pred_nu = scaler_y.inverse_transform(pred_nu_scaled.reshape(-1,1)).flatten()
        
    thresh_nu = get_max_gap_threshold(w_nu)
    sel_nu = np.where(w_nu > thresh_nu)[0]
    
    # --- 模型 2: Lasso ---
    print("\n[2/4] Training Lasso...")
    lasso = LassoCV(cv=5, random_state=cfg["seed"]).fit(X_train_s, y_train_s)
    w_lasso = np.abs(lasso.coef_)
    pred_lasso_scaled = lasso.predict(X_test_s)
    pred_lasso = scaler_y.inverse_transform(pred_lasso_scaled.reshape(-1,1)).flatten()
    sel_lasso = np.where(w_lasso > cfg["lasso_threshold"])[0]
    
    # --- 模型 3: Random Forest ---
    print("\n[3/4] Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=cfg["rf_estimators"], random_state=cfg["seed"])
    rf.fit(X_train_s, y_train) # RF 可以直接训原始 y
    w_rf = rf.feature_importances_
    pred_rf = rf.predict(X_test_s)
    thresh_rf = get_max_gap_threshold(w_rf)
    sel_rf = np.where(w_rf > thresh_rf)[0]
    
    # --- 模型 4: GAMs (Spline + LR) ---
    print("\n[4/4] Training GAMs (Spline)...")
    # 为了简化，我们只计算 MSE，不进行复杂的特征选择（GAM 特征选择比较耗时）
    gam_pipe = make_pipeline(
        SplineTransformer(n_knots=5, degree=3),
        LinearRegression()
    )
    gam_pipe.fit(X_train_s, y_train)
    pred_gam = gam_pipe.predict(X_test_s)
    sel_gam = [] # 暂不评估 GAM 的特征选择
    
    # --- 结果汇总 ---
    metrics = {
        "Neural UniLasso": calculate_metrics(y_test, pred_nu, sel_nu, true_idx, cfg["p_features"]),
        "Lasso": calculate_metrics(y_test, pred_lasso, sel_lasso, true_idx, cfg["p_features"]),
        "Random Forest": calculate_metrics(y_test, pred_rf, sel_rf, true_idx, cfg["p_features"]),
        "GAMs": {"MSE": mean_squared_error(y_test, pred_gam), "F1": 0, "Sensitivity": 0, "Specificity": 0, "Selected_Num": 0}
    }
    
    # --- 文本输出 ---
    print("\n" + "="*60)
    print(f"{'Method':<20} | {'Selected Features':<30}")
    print("-" * 60)
    print(f"{'Neural UniLasso':<20} | {str(sel_nu)}")
    print(f"{'Lasso':<20} | {str(sel_lasso)}")
    print(f"{'Random Forest':<20} | {str(sel_rf)}")
    print("="*60)
    
    result_df = pd.DataFrame(metrics).T
    print("\nExperimental Metrics:")
    print(result_df)

    # ==========================================
    # 5. 可视化 (三图流)
    # ==========================================
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle("Neural UniLasso: Comprehensive Experiment Results", fontsize=22, y=0.96)
    
    # --- 图一：特征选择对比 (柱状图) ---
    ax1 = plt.subplot(2, 2, 1)
    disp_p = 20 # 只看前20个
    idx = np.arange(disp_p)
    width = 0.25
    
    # 归一化函数
    def norm(x): return x / x.max() if x.max() > 0 else x
    
    ax1.bar(idx - width, norm(w_nu)[:disp_p], width, label='Neural UniLasso', color='#3498db')
    ax1.bar(idx, norm(w_lasso)[:disp_p], width, label='Lasso', color='#e74c3c', alpha=0.7)
    ax1.bar(idx + width, norm(w_rf)[:disp_p], width, label='Random Forest', color='#2ecc71', alpha=0.7)
    
    for i in true_idx:
        ax1.axvline(i, color='purple', linestyle='--', linewidth=1.5)
        ax1.text(i, 1.02, f'X{i+1}', ha='center', color='purple', fontweight='bold', transform=ax1.get_xaxis_transform())
        
    ax1.set_title("Feature Importance Comparison (Top 20 Features)", fontsize=14)
    ax1.set_xlabel("Feature Index")
    ax1.set_ylabel("Normalized Importance")
    ax1.legend()
    
    # --- 图二：单变量特征学习 (Shape Recovery) ---
    ax2 = plt.subplot(2, 2, 2)
    # 构造测试区间 X_grid
    x_raw = np.linspace(-3, 3, 100)
    # 必须标准化后输入网络
    x_input = scaler_X.transform(np.zeros((100, cfg["p_features"]))) # Dummy
    # 填充第一列为测试值 (假设标准化对每列都差不多，因为是均匀分布)
    x_input_col = (x_raw - scaler_X.mean_[0]) / scaler_X.scale_[0]
    x_tensor = torch.FloatTensor(x_input_col).view(-1, 1)
    
    colors = ['#e67e22', '#9b59b6', '#34495e', '#16a085']
    funcs = ["sin(x1)", "-3*tan(x2)", "5*e^x3", "-2*x4"]
    
    for i, feat_id in enumerate(true_idx):
        # 1. 网络学习到的形状
        with torch.no_grad():
            z = nu_model.uni_nets[feat_id](x_tensor).numpy().flatten()
            contrib = z * w_nu[feat_id] # 乘以权重
        
        # 2. 真实形状
        if feat_id == 0: y_true = np.sin(x_raw)
        elif feat_id == 1: y_true = -3 * np.clip(np.tan(x_raw), -10, 10)
        elif feat_id == 2: y_true = 5 * np.exp(x_raw)
        elif feat_id == 3: y_true = -2 * x_raw
        
        # 3. 归一化到 [0,1] 区间以便仅比较“形状”
        def norm_shape(arr): 
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        
        offset = i * 1.2
        ax2.plot(x_raw, norm_shape(contrib) + offset, color=colors[i], lw=3, label=f'Learned X{feat_id+1}')
        ax2.plot(x_raw, norm_shape(y_true) + offset, color='black', ls=':', lw=1.5, label='True' if i==0 else "")
        ax2.text(x_raw[0], 0.5 + offset, funcs[i], fontsize=10, color=colors[i], fontweight='bold')
        
    ax2.set_title("Learned Shapes vs True Shapes (Normalized & Offset)", fontsize=14)
    ax2.set_xlabel("Input X value")
    ax2.set_yticks([])
    ax2.legend(loc='upper right')
    
    # --- 图三：指标对比 (雷达图/柱状图) ---
    ax3 = plt.subplot(2, 1, 2)
    
    # 准备数据
    plot_df = result_df.drop("GAMs") # GAMs 没有分类指标
    plot_models = plot_df.index
    x = np.arange(len(plot_models))
    width = 0.2
    
    # 双轴：左边 F1/Sens/Spec, 右边 MSE
    rects1 = ax3.bar(x - width, plot_df["F1"], width, label='F1 Score', color='#f1c40f')
    rects2 = ax3.bar(x, plot_df["Sensitivity"], width, label='Sensitivity', color='#e67e22')
    rects3 = ax3.bar(x + width, plot_df["Specificity"], width, label='Specificity', color='#95a5a6')
    
    ax3.set_ylabel('Score (0-1)')
    ax3.set_title('Performance Metrics Comparison', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(plot_models)
    ax3.set_ylim(0, 1.15)
    ax3.legend(loc='upper left')
    
    # 右轴画 MSE
    ax4 = ax3.twinx()
    # 稍微偏移一点避免重叠，或者单独画
    mse_vals = plot_df["MSE"]
    ax4.plot(x, mse_vals, color='#2c3e50', marker='o', linewidth=2, linestyle='--', label='MSE (Line)')
    ax4.set_ylabel('MSE (Lower is Better)')
    ax4.legend(loc='upper right')
    
    # 标注数值
    for rect in rects1 + rects2 + rects3:
        h = rect.get_height()
        ax3.text(rect.get_x()+rect.get_width()/2, h, f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()