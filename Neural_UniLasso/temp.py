import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, f1_score, recall_score, precision_score

# ==========================================
# 0. 实验配置表 (Experiment Configuration)
# ==========================================
CONFIG = {
    # -- 硬件控制 --
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    
    # -- 数据生成 --
    "n_train": 1000,
    "n_test": 2000,
    "p_features": 200,
    "data_range": (-3, 3),    # U(-3, 3)
    "noise_std": 0.5,         # 噪声水平
    
    # -- 神经网络架构 --
    "hidden_structure": [16, 8], 
    "activation": "ELU",
    
    # -- 训练参数 --
    "epochs": 1500,
    "lr": 0.01,
    "batch_size": 256,       # 使用 Batch 训练加速
    "l1_lambda": 0.05,       # 稀疏惩罚系数
    
    # -- 绘图参数 --
    "vis_x_range": (-4, 4),  # 形状可视化范围（比训练范围稍大，看泛化）
    "seed": 42
}

# 全局设置
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")
DEVICE = torch.device(CONFIG["device"])

print(f"🚀 Experiment running on device: {DEVICE}")

# ==========================================
# 1. 核心模型：Neural UniLasso (GPU Optimized)
# ==========================================
class UnivariateNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        h1, h2 = CONFIG["hidden_structure"]
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
        # 融合权重
        self.theta = nn.Parameter(torch.rand(n_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x: [batch, p]
        z_list = []
        for i in range(self.n_features):
            z_list.append(self.uni_nets[i](x[:, i].view(-1, 1)))
        
        Z = torch.cat(z_list, dim=1) 
        
        # 非负约束
        self.weights = F.softplus(self.theta)
        
        y_pred = torch.matmul(Z, self.weights) + self.bias
        return y_pred, self.weights, Z

# ==========================================
# 2. 数据生成器 (含复杂函数处理)
# ==========================================
def generate_data():
    n = CONFIG["n_train"] + CONFIG["n_test"]
    p = CONFIG["p_features"]
    low, high = CONFIG["data_range"]
    
    X = np.random.uniform(low, high, size=(n, p))
    
    # 提取真实变量
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    
    # y = sin(x1) - 3*tan(x2) + 5*e^x3 - 2*x4
    # 注意：tan(x) 在 (-3,3) 内有奇点 +/- 1.57。如果不截断，y会由极大值主导。
    t1 = np.sin(x1)
    t2 = -3 * np.clip(np.tan(x2), -10, 10) # 截断 tan
    t3 = 5 * np.exp(x3)                    # e^3 ~ 20, e^-3 ~ 0
    t4 = -2 * x4
    
    y_raw = t1 + t2 + t3 + t4
    y = y_raw + np.random.normal(0, CONFIG["noise_std"], n)
    
    true_idx = [0, 1, 2, 3]
    
    return X[:CONFIG["n_train"]], X[CONFIG["n_train"]:], \
           y[:CONFIG["n_train"]], y[CONFIG["n_train"]:], \
           true_idx

# ==========================================
# 3. 智能阈值截断 (Smart Cliff Detection)
# ==========================================
def get_cliff_threshold(weights):
    """
    寻找权重排序后的最大断崖，作为信号与噪声的分界线。
    """
    w_abs = np.abs(weights)
    sorted_w = np.sort(w_abs)[::-1] # 降序
    
    # 仅在头部区域搜索 (假设真实变量是少数)
    search_len = min(len(sorted_w)-1, int(len(sorted_w)*0.2) + 5)
    
    # 计算相邻落差
    gaps = sorted_w[:search_len] - sorted_w[1:search_len+1]
    
    if len(gaps) == 0: return 0.0
    
    best_gap_idx = np.argmax(gaps)
    
    # 阈值取断崖中间
    threshold = (sorted_w[best_gap_idx] + sorted_w[best_gap_idx+1]) / 2
    return threshold

# ==========================================
# 4. 指标计算器
# ==========================================
def calc_metrics(y_true, y_pred, selected, true_idx, p):
    mse = mean_squared_error(y_true, y_pred)
    
    y_true_bin = np.zeros(p)
    y_true_bin[true_idx] = 1
    
    y_pred_bin = np.zeros(p)
    y_pred_bin[selected] = 1
    
    # 指标
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    sens = recall_score(y_true_bin, y_pred_bin, zero_division=0) # Sensitivity/Recall
    
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    spec = tn / (tn + fp) if (tn+fp) > 0 else 0.0 # Specificity
    
    return {"MSE": mse, "F1": f1, "Sensitivity": sens, "Specificity": spec, "Count": len(selected)}

# ==========================================
# 5. 主实验流程
# ==========================================
def run_experiment():
    print(f"\n{'='*20} Experiment Start {'='*20}")
    print(f"Config: N={CONFIG['n_train']}, P={CONFIG['p_features']}, Device={DEVICE}")
    
    # 1. 数据准备
    X_train, X_test, y_train, y_test, true_idx = generate_data()
    
    # 标准化 (Neural Net 需要输入标准化，GAM/Lasso 也受益)
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    
    # y 也要标准化以稳定梯度，但在评估 MSE 时需还原
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # ----------------------------------------------------
    # Model A: Neural UniLasso (GPU)
    # ----------------------------------------------------
    print("\n[Model 1] Training Neural UniLasso...")
    model = NeuralUniLasso(CONFIG["p_features"]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    loss_fn = nn.MSELoss()
    
    # 转为 GPU Tensor
    Xt = torch.tensor(X_train_s, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train_s, dtype=torch.float32, device=DEVICE)
    
    # 训练循环 (带进度条)
    loop = tqdm(range(CONFIG["epochs"]), desc="Training")
    for epoch in loop:
        opt.zero_grad()
        pred, w, _ = model(Xt)
        
        mse = loss_fn(pred.view(-1), yt)
        reg = CONFIG["l1_lambda"] * torch.sum(w)
        loss = mse + reg
        
        loss.backward()
        opt.step()
        
        if epoch % 50 == 0:
            loop.set_postfix(loss=loss.item(), active=(w>1e-3).sum().item())
            
    # 推理
    model.eval()
    Xt_test = torch.tensor(X_test_s, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        _, w_tensor, _ = model(Xt_test)
        w_nu = w_tensor.cpu().numpy()
        pred_scaled = model(Xt_test)[0].cpu().numpy().flatten()
        pred_nu = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
        
    # 智能截断
    thresh_nu = get_cliff_threshold(w_nu)
    sel_nu = np.where(w_nu > thresh_nu)[0]
    
    # ----------------------------------------------------
    # Model B: Lasso
    # ----------------------------------------------------
    print("[Model 2] Training LassoCV...")
    lasso = LassoCV(cv=5, n_jobs=-1).fit(X_train_s, y_train_s)
    w_lasso = np.abs(lasso.coef_)
    pred_lasso = scaler_y.inverse_transform(lasso.predict(X_test_s).reshape(-1,1)).flatten()
    sel_lasso = np.where(w_lasso > 1e-4)[0]
    
    # ----------------------------------------------------
    # Model C: Random Forest
    # ----------------------------------------------------
    print("[Model 3] Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1).fit(X_train_s, y_train)
    w_rf = rf.feature_importances_
    pred_rf = rf.predict(X_test_s)
    thresh_rf = get_cliff_threshold(w_rf)
    sel_rf = np.where(w_rf > thresh_rf)[0]
    
    # ----------------------------------------------------
    # Model D: GAMs (Simplified)
    # ----------------------------------------------------
    print("[Model 4] Training GAMs (Splines)...")
    # 为了特征选择，我们对 Spline 系数进行组 L1 范数筛选比较困难
    # 这里我们只计算 GAM 的预测 MSE 作为非线性模型的基准
    gam = make_pipeline(SplineTransformer(n_knots=5, degree=3), LinearRegression())
    gam.fit(X_train_s, y_train)
    pred_gam = gam.predict(X_test_s)
    sel_gam = [] # 暂不参与特征选择对比
    
    # ----------------------------------------------------
    # 结果汇总
    # ----------------------------------------------------
    res = {
        "Neural UniLasso": calc_metrics(y_test, pred_nu, sel_nu, true_idx, CONFIG["p_features"]),
        "Lasso": calc_metrics(y_test, pred_lasso, sel_lasso, true_idx, CONFIG["p_features"]),
        "Random Forest": calc_metrics(y_test, pred_rf, sel_rf, true_idx, CONFIG["p_features"]),
        "GAMs": {"MSE": mean_squared_error(y_test, pred_gam), "F1":0, "Sensitivity":0, "Specificity":0, "Count":0}
    }
    
    df_res = pd.DataFrame(res).T
    print("\n" + "="*40)
    print("EXPERIMENTAL RESULTS")
    print("="*40)
    print(f"Neural Selected: {sel_nu}")
    print(f"Lasso Selected:  {sel_lasso}")
    print(df_res)
    
    # ==========================================
    # 6. 可视化
    # ==========================================
    plot_visualization(w_nu, w_lasso, w_rf, true_idx, model, scaler_X, scaler_y, df_res)

def plot_visualization(w_nu, w_lasso, w_rf, true_idx, model, scaler_X, scaler_y, df_res):
    fig = plt.figure(figsize=(20, 16))
    plt.suptitle("Neural UniLasso: Feature Learning Benchmark", fontsize=24, y=0.96)
    
    # --- 图一：特征选择对比 ---
    ax1 = plt.subplot(2, 2, 1)
    disp_p = 20
    idx = np.arange(disp_p)
    
    def norm(x): return x/x.max() if x.max()>0 else x
    
    ax1.bar(idx-0.2, norm(w_nu)[:disp_p], 0.2, label='Neural UniLasso', color='#2980b9')
    ax1.bar(idx, norm(w_lasso)[:disp_p], 0.2, label='Lasso', color='#e74c3c', alpha=0.7)
    ax1.bar(idx+0.2, norm(w_rf)[:disp_p], 0.2, label='Random Forest', color='#27ae60', alpha=0.7)
    
    for i in true_idx:
        ax1.axvline(i, color='purple', ls='--', lw=2)
        ax1.text(i, 1.05, f'True X{i+1}', ha='center', color='purple', fontweight='bold')
    
    ax1.set_title("Feature Importance Ranking (Top 20)", fontsize=16)
    ax1.set_xlabel("Feature Index")
    ax1.legend()

    # --- 图二：单变量形状学习 (不压缩范围，看真实拟合) ---
    ax2 = plt.subplot(2, 2, 2)
    
    # 生成宽范围测试数据 (-4, 4)，看外推能力
    x_viz = np.linspace(CONFIG["vis_x_range"][0], CONFIG["vis_x_range"][1], 200)
    # 构造输入 (填充到对应列)
    x_in = np.zeros((200, CONFIG["p_features"]))
    # 简单假设所有列均值方差相似(均匀分布特性)，直接用 fit 时的 scaler 变换
    # 实际应针对每一列，但这里数据同分布，直接用 transform 的参数
    x_in_s = scaler_X.transform(x_in) 
    # 替换前几列为我们的 x_viz (标准化后的)
    x_viz_s = (x_viz - scaler_X.mean_[0]) / scaler_X.scale_[0]
    
    colors = ['#d35400', '#8e44ad', '#2c3e50', '#16a085']
    labels = ["sin(x1)", "-3*tan(x2)", "5*e^x3", "-2*x4"]
    
    for i, feat_id in enumerate(true_idx):
        # 1. 神经网络输出
        xt = torch.tensor(x_viz_s, dtype=torch.float32, device=DEVICE).view(-1, 1)
        with torch.no_grad():
            z = model.uni_nets[feat_id](xt).cpu().numpy().flatten()
            # 有效贡献 = z * theta (尚未反归一化)
            eff_contrib_s = z * w_nu[feat_id]
            # 反归一化到 y 的原始尺度
            eff_contrib = eff_contrib_s * scaler_y.scale_[0] 
            # 忽略 bias，因为形状主要由权重决定，bias 是全局的
            
        # 2. 真实函数
        if feat_id == 0: y_true = np.sin(x_viz)
        elif feat_id == 1: y_true = -3 * np.tan(x_viz) # 画图时不截断，看区别，或者截断
        elif feat_id == 2: y_true = 5 * np.exp(x_viz)
        elif feat_id == 3: y_true = -2 * x_viz
        
        # 处理 y_true 的 infinite (为了画图美观)
        y_true = np.clip(y_true, -50, 50)
        
        ax2.plot(x_viz, eff_contrib, color=colors[i], lw=3, label=f'Learned X{feat_id+1}')
        ax2.plot(x_viz, y_true, color=colors[i], ls=':', lw=1.5, alpha=0.6)
    
    ax2.set_title("Learned Function Shapes (Effective Contribution)", fontsize=16)
    ax2.set_xlabel("Input X (Original Scale)")
    ax2.set_ylabel("Contribution to Y")
    ax2.set_ylim(-30, 30) # 限制 Y 轴看清细节
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- 图三：指标雷达/柱状图 ---
    ax3 = plt.subplot(2, 1, 2)
    df_plot = df_res.drop("GAMs") # GAM 无分类指标
    x = np.arange(len(df_plot))
    w = 0.2
    
    ax3.bar(x-w, df_plot["F1"], w, label='F1 Score', color='#f1c40f')
    ax3.bar(x, df_plot["Sensitivity"], w, label='Sensitivity', color='#e67e22')
    ax3.bar(x+w, df_plot["Specificity"], w, label='Specificity', color='#95a5a6')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_plot.index, fontsize=12)
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Performance Metrics", fontsize=16)
    ax3.legend(loc='upper left')
    
    # 双轴画 MSE
    ax4 = ax3.twinx()
    ax4.plot(x, df_plot["MSE"], color='#2c3e50', marker='D', ms=10, lw=2, label='MSE')
    ax4.set_ylabel("MSE (Lower is Better)")
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
