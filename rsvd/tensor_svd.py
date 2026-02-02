from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.fft
from dataclasses import dataclass

# ==========================================
# 1. 系统配置
# ==========================================
@dataclass
class Config:
    tensor_inner: int = 10       # 内部迭代次数
    lamb: float = 0.0005         # 基础正则化系数 lambda
    p: float = 0.75              # Lp 范数 (0 < p <= 1)
    epsilon: float = 1e-10       # 防止除零的微小量

# ==========================================
# 2. 核心数学算子 (Lp 最小化求解器)
# ==========================================
def solve_lp_w(y: torch.Tensor, weight_vec: torch.Tensor, p: float, inner_iter: int = 4) -> torch.Tensor:
    """
    求解加权 Lp 范数最小化问题的近端算子。
    """
    # 1. 计算自适应阈值 tau
    a = 2 * weight_vec * (1 - p)
    power_factor = 1 / (2 - p)
    
    # weight_vec 已经是 lambda * w
    tau = torch.pow(a, power_factor) + p * weight_vec * torch.pow(a, (p - 1) * power_factor)
    
    # 2. 掩码筛选
    mask = torch.abs(y) > tau
    x = torch.zeros_like(y)
    
    if not mask.any():
        return x
    
    # 3. 广义软阈值迭代 (GST)
    y0 = y[mask]
    lambda0 = weight_vec[mask]
    t = torch.abs(y0)
    
    for _ in range(inner_iter):
        t = torch.abs(y0) - p * lambda0 * torch.pow(t, p - 1)
    
    x[mask] = torch.sign(y0) * t
    return x

# ==========================================
# 3. 主算法流程 (针对 Batch 为变换维)
# ==========================================
def tensor_log_sp(Y: torch.Tensor, lambdai: float, par: Config) -> torch.Tensor:
    """
    输入形状: (Batch, Length, Width)
    逻辑:
      - 将 Batch (dim 0) 视为变换维度 (Transform Dim / Tube)
      - 将 (Length, Width) 视为矩阵切片
    """
    # Y shape: (B, L, W)
    B, L, W = Y.shape
    
    # -----------------------------------------------------------
    # Step 1: FFT 变换 (沿着 Batch 维度)
    # -----------------------------------------------------------
    # 直接沿着 dim=0 进行 FFT
    # Y_f shape: (B, L, W) - 复数张量
    Y_f = torch.fft.fft(Y, dim=0)
    
    # 数据清洗
    Y_f = torch.nan_to_num(Y_f, nan=0.0, posinf=0.0, neginf=0.0)

    # -----------------------------------------------------------
    # Step 2: 并行 SVD 分解
    # -----------------------------------------------------------
    # PyTorch 的 svd 默认处理最后两个维度作为矩阵，前面的维度作为 Batch。
    # 我们的形状是 (B, L, W)，这完美符合要求！
    # 含义：对频域中的每一个 'B' (频率点)，分解其 (L, W) 矩阵。
    
    # U_f: (B, L, K), S_f: (B, K), Vh_f: (B, K, W)
    # 其中 K = min(L, W)
    U_f, S_f, Vh_f = torch.linalg.svd(Y_f, full_matrices=False)
    
    # -----------------------------------------------------------
    # Step 3: 迭代加权收缩
    # -----------------------------------------------------------
    # S_f 是奇异值 (实数)
    w = 1.0 / (torch.pow(S_f, par.p) + par.epsilon)
    s1 = torch.zeros_like(S_f)
    
    for _ in range(par.tensor_inner):
        w_vec = lambdai * w
        # 调用求解器 (广播机制会自动处理 Batch 维)
        s1 = solve_lp_w(S_f, w_vec, par.p)
        # 更新权重
        w = 1.0 / (torch.pow(s1, par.p) + par.epsilon)

    # -----------------------------------------------------------
    # Step 4: 频域重构
    # -----------------------------------------------------------
    # 构造对角矩阵: (B, K) -> (B, K, K)
    S_diag = torch.diag_embed(s1)
    
    # [关键修复]: 类型匹配
    # 将实数对角阵转为复数，以便与 U_f (复数) 相乘
    S_diag = S_diag.to(U_f.dtype)
    
    # 矩阵乘法: (B, L, K) @ (B, K, K) @ (B, K, W) -> (B, L, W)
    # 这里不需要 permute，因为维度顺序已经是正确的
    X_f = U_f @ S_diag @ Vh_f
    
    # -----------------------------------------------------------
    # Step 5: 逆变换
    # -----------------------------------------------------------
    # 沿着 Batch 维度 (dim=0) 逆 FFT
    TensorX = torch.fft.ifft(X_f, dim=0)
    
    # 取实部
    X = torch.real(TensorX)
    
    return X

# ==========================================
# 2. 实验工具函数
# ==========================================
def generate_low_rank_data(batch, length, width, rank=5):
    """
    生成一个人造的低秩张量。
    原理：通过两个小矩阵相乘 (L, r) * (r, W) 生成秩为 r 的矩阵，并扩展到 Batch。
    """
    torch.manual_seed(1024) # 固定随机种子以便复现
    
    # 我们生成随 Batch 缓慢变化的低秩矩阵，模拟视频或时间序列
    data = []
    # 基础矩阵
    U = torch.randn(length, rank)
    V = torch.randn(rank, width)
    
    for i in range(batch):
        # 对每一帧加一点微小的扰动，保持整体相关性，但又不完全相同
        Ui = U + 0.1 * torch.randn_like(U) * np.sin(i/5.0)
        Vi = V + 0.1 * torch.randn_like(V) * np.cos(i/5.0)
        data.append(Ui @ Vi)
        
    return torch.stack(data) # (Batch, Length, Width)

def calc_psnr(clean, recovered):
    """计算峰值信噪比 (PSNR)，图像处理标准指标"""
    mse = torch.mean((clean - recovered) ** 2)
    if mse == 0: return float('inf')
    max_pixel = clean.max()
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

# ==========================================
# 3. 运行主实验
# ==========================================
def run_demo():
    # 设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在设备 {device} 上启动测试...\n")
    
    # 参数
    B, L, W = 32, 64, 64  # 批次(时间), 长, 宽
    RANK = 5              # 真实的秩
    NOISE_LEVEL = 0.4     # 噪声强度
    
    # 1. 制造数据
    print("Step 1: 生成低秩真值数据...")
    clean_tensor = generate_low_rank_data(B, L, W, RANK).to(device)
    
    # 2. 添加噪声
    print("Step 2: 添加高斯噪声...")
    noise = torch.randn_like(clean_tensor) * NOISE_LEVEL * clean_tensor.std()
    noisy_tensor = clean_tensor + noise
    
    # 3. 运行算法
    print("Step 3: 运行 Tensor Log-Sp 算法 (这可能需要几秒钟)...")
    config = Config(tensor_inner=10, p=0.75, lamb=0.0005) # 参数配置
    
    # 注意：lambda 需要根据噪声水平微调，这里给一个经验值
    # 噪声越大，lambda 应该稍微大一点来增强过滤
    algo_lambda = 0.05 * NOISE_LEVEL 
    
    recovered_tensor = tensor_log_sp(noisy_tensor, algo_lambda, config)
    
    # 4. 评估结果
    psnr_noisy = calc_psnr(clean_tensor, noisy_tensor)
    psnr_recovered = calc_psnr(clean_tensor, recovered_tensor)
    
    print("\n" + "="*40)
    print(f"📊 实验结果报告")
    print("="*40)
    print(f"噪声图像 PSNR: {psnr_noisy:.2f} dB (越低越差)")
    print(f"恢复图像 PSNR: {psnr_recovered:.2f} dB (越高越好)")
    print(f"提升幅度: +{psnr_recovered - psnr_noisy:.2f} dB")
    print("="*40 + "\n")
    
    # 5. 可视化绘图
    plot_results(clean_tensor, noisy_tensor, recovered_tensor, idx=B//2)
    analyze_rank_recovery(clean_tensor, noisy_tensor, recovered_tensor)

def plot_results(clean, noisy, recovered, idx):
    """绘制对比图：选取 Batch 中的某一帧进行展示"""
    clean_img = clean[idx].cpu().numpy()
    noisy_img = noisy[idx].cpu().numpy()
    rec_img = recovered[idx].cpu().detach().numpy()
    
    plt.figure(figsize=(15, 5))
    
    # 设置统一的色阶范围，方便对比
    vmin, vmax = clean_img.min(), clean_img.max()
    
    plt.subplot(1, 3, 1)
    plt.title("Original (Clean Low-Rank)")
    plt.imshow(clean_img, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Corrupted (Input)")
    plt.imshow(noisy_img, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Recovered (Algorithm Output)")
    plt.imshow(rec_img, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"已展示第 {idx} 帧的切片对比。")
    

def analyze_rank_recovery(clean, noisy, recovered):
    """
    分析并绘制奇异值分布谱 (Singular Value Spectrum)。
    我们在频域计算 SVD，这是算法实际工作的地方。
    """
    import matplotlib.pyplot as plt
    
    # 辅助函数：计算频域平均奇异值
    def get_singular_spectrum(tensor):
        # 1. FFT 变换 (沿着 Batch 维度)
        tensor_f = torch.fft.fft(tensor, dim=0)
        
        # 2. SVD 分解
        # PyTorch SVD 自动处理最后两维 (L, W)
        # S_f shape: (Batch, K) where K = min(L, W)
        _, S_f, _ = torch.linalg.svd(tensor_f, full_matrices=False)
        
        # 3. 计算所有频率切片的平均奇异值
        # 这代表了张量的平均能量分布
        mean_singular_values = torch.mean(S_f, dim=0).cpu().detach().numpy()
        
        # 归一化 (让最大值为 1，方便对比形状)
        return mean_singular_values / mean_singular_values[0]

    # 获取三组数据的谱
    s_clean = get_singular_spectrum(clean)
    s_noisy = get_singular_spectrum(noisy)
    s_rec = get_singular_spectrum(recovered)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    x_axis = range(len(s_clean))
    
    # 使用对数坐标，因为奇异值下降非常快，对数轴能看清细节
    plt.semilogy(x_axis, s_clean, 'g-', linewidth=2, label='Ground Truth (Low Rank)')
    plt.semilogy(x_axis, s_noisy, 'r--', linewidth=1.5, alpha=0.6, label='Noisy Input (Long Tail)')
    plt.semilogy(x_axis, s_rec, 'b.-', linewidth=2, label='Recovered (Algorithm)')
    
    plt.title("Singular Value Spectrum Analysis (Log Scale)")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Magnitude (Log)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # 标注真实的秩 (假设生成数据时 rank=5)
    plt.axvline(x=5, color='k', linestyle=':', label='True Rank Cutoff')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()
    