# 创新点1：更换一个投影算子

import torch
import torch.fft

def solve_lp_w(y, lambda_val, p, J=4):
    """
    广义软阈值 (GST) 算子 - 保持不变
    求解: min_x 0.5(y-x)^2 + lambda * |x|^p
    """
    # 计算阈值 tau
    a = 2 * lambda_val * (1 - p)
    # 加上 1e-16 防止底数为 0 导致梯度或计算异常
    power_term = (p - 1) / (2 - p)
    tau = torch.pow(a, 1 / (2 - p)) + p * lambda_val * torch.pow(a, power_term)

    x = torch.zeros_like(y)
    
    # 找到绝对值大于阈值的索引
    mask = torch.abs(y) > tau
    
    if mask.sum() > 0:
        y0 = y[mask]
        lambda0 = lambda_val if isinstance(lambda_val, float) else lambda_val[mask]
        
        # 不动点迭代初始化
        t = torch.abs(y0)
        
        # 迭代逼近最优解
        for _ in range(J):
            t = torch.abs(y0) - p * lambda0 * torch.pow(t, p - 1)
        
        x[mask] = torch.sign(y0) * t
        
    return x

def tensor_log_sp_channel_first(Y, lambdai, par):
    """
    针对 (Channel, Width, Length) 输入的低秩张量恢复算法
    
    逻辑:
    1. 沿 Channel (dim=0) 做 FFT
    2. 对每个频率切片 (Width, Length) 做并行 SVD
    3. 收缩奇异值
    4. 沿 Channel (dim=0) 做 IFFT
    """
    # 获取参数
    p = par.get('p', 0.75)
    inner_iter = par.get('tensorinner', 10)
    epsilon = par.get('epsilon', 1e-10)
    
    # Y shape: (C, W, L)
    
    # --- 1. FFT 变换 (沿 Channel 维度) ---
    # PyTorch FFT 默认输出复数张量
    # dim=0 对应 Channel
    yf = torch.fft.fft(Y, dim=0)
    
    # yf shape 依然是 (C, W, L)
    # 在 t-SVD 语境下，这里的 C 个切片即为频域下的正面切片 (Frontal Slices)
    
    # 处理数值稳定性
    #yf = torch.nan_to_num(yf, nan=0.0, posinf=0.0, neginf=0.0)
    yf_real = torch.nan_to_num(yf.real, nan=0.0, posinf=0.0, neginf=0.0)
    yf_imag = torch.nan_to_num(yf.imag, nan=0.0, posinf=0.0, neginf=0.0)
    yf = torch.complex(yf_real, yf_imag)

    # --- 2. 并行 SVD 分解 ---
    # PyTorch 的 linalg.svd 输入为 (..., M, N)
    # 它会将前面的维度 (这里是 C) 作为 batch 处理
    # 对 C 个 (W, L) 矩阵同时做 SVD，完全并行
    # full_matrices=False 对应 MATLAB 的 'econ'
    u, s, vh = torch.linalg.svd(yf, full_matrices=False)
    
    # u:  (C, W, K)
    # s:  (C, K)      <-- 奇异值向量
    # vh: (C, K, L)
    # 其中 K = min(W, L)

    # --- 3. 迭代加权收缩 (Iterative Reweighting) ---
    
    # 初始化权重 w (利用广播机制)
    w = 1.0 / (torch.pow(s, p) + epsilon)
    s1 = torch.zeros_like(s)
    
    for _ in range(inner_iter):
        # 计算加权惩罚向量
        w_vec = lambdai * w
        
        # 执行 GST 收缩 (这是 element-wise 操作，天生支持并行)
        s1 = solve_lp_w(s, w_vec, p)
        
        # 更新权重
        w = 1.0 / (torch.pow(s1, p) + epsilon)
        
    # --- 4. 频域重构 ---
    
    # 将收缩后的奇异值向量 s1 转回对角矩阵
    # s1 shape: (C, K) -> s_mat shape: (C, K, K)
    s_mat = torch.diag_embed(s1).to(yf.dtype) # 必须转为复数类型
    
    # 矩阵乘法重构 Xf = U * S * Vh
    # PyTorch 的 @ 运算符支持 batch matmul
    # (C,W,K) @ (C,K,K) -> (C,W,K)
    # (C,W,K) @ (C,K,L) -> (C,W,L)
    xf = u @ s_mat @ vh
    
    # --- 5. IFFT 逆变换 ---
    
    # 沿 Channel (dim=0) 逆变换
    tensor_x = torch.fft.ifft(xf, dim=0)
    
    # 取实部 (恢复原始数据域)
    X = torch.real(tensor_x)
    
    return X

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 模拟输入数据 (Channel, Width, Length)
    # 例如：64个通道，图像大小 100x100
    C, W, L = 64, 100, 100
    
    # 2. 构造低秩数据 (用于验证)
    # 在每个矩阵切片上构造低秩，并在通道间保持某种相关性
    r = 10 # 秩
    U_true = torch.randn(C, W, r, device=device)
    V_true = torch.randn(C, r, L, device=device)
    # 这是一个简单的模拟，真实的t-SVD低秩是在t-product意义下的
    Y_clean = U_true @ V_true 
    
    # 添加噪声
    noise = 0.05 * torch.randn(C, W, L, device=device)
    Y_noisy = Y_clean + noise
    
    # 3. 参数设置
    par = {
        'tensorinner': 5,   # 内部迭代次数
        'p': 0.8,           # Lp 范数 p 值
        'epsilon': 1e-8
    }
    lambda_val = 0.01       # 正则化强度，需根据噪声大小调整

    print(f"Input Tensor Shape: {Y_noisy.shape} (Channel, Width, Length)")
    print(f"Processing on: {device}")

    # 4. 运行算法
    X_recovered = tensor_log_sp_channel_first(Y_noisy, lambda_val, par)

    print("Done.")
    print(f"Recovered Shape: {X_recovered.shape}")
    
    # 计算相对误差
    err = torch.norm(X_recovered - Y_clean) / torch.norm(Y_clean)
    print(f"Relative Recovery Error: {err.item():.5f}")