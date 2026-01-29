# 文件名: randomized_svd.py

import numpy as np
import torch
import time
from typing import Tuple, Union

# --- 模块化导入 ---
try:
    from adaptive_range_finder import adaptive_randomized_range_finder
except ImportError:
    print("错误: 找不到 core_utils.py。请确保上一段代码已保存为该文件名。")
    exit(1)

def _randomized_svd_2d_padded(
    A: np.ndarray, 
    epsilon: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    内部函数：执行自适应 SVD，然后将结果强制填充(Padding)到原始维度。
    """
    m, n = A.shape
    min_dim = min(m, n) # 对于 64x64，这就是 64
    
    # 1. 自适应计算 (得到的 k 可能远小于 64)
    Q = adaptive_randomized_range_finder(A, epsilon=epsilon)
    B = Q.T @ A
    S_hat, Sigma_small, Vt_small = np.linalg.svd(B, full_matrices=False)
    U_small = Q @ S_hat
    
    # 获取当前的有效秩 k
    k = Sigma_small.shape[0]
    
    # 如果算法发现 k >= min_dim，说明不需要补全，或者容差设置得太小
    if k >= min_dim:
        # 为了安全，截断到 min_dim
        return U_small[:, :min_dim], Sigma_small[:min_dim], Vt_small[:min_dim, :]
    
    # 2. 执行零填充 (Zero Padding)
    # 目标: S变成 (min_dim,), U变成 (m, min_dim), Vt变成 (min_dim, n)
    
    # --- 补全 S ---
    Sigma_final = np.zeros(min_dim, dtype=A.dtype)
    Sigma_final[:k] = Sigma_small # 前 k 个填入真实值，后面全是 0
    
    # --- 补全 U ---
    # U 的形状通常是 (m, k)，我们需要把它变成 (m, min_dim)
    # 我们在右边增加 (min_dim - k) 列的 0
    U_final = np.zeros((m, min_dim), dtype=A.dtype)
    U_final[:, :k] = U_small
    
    # --- 补全 Vt ---
    # Vt 的形状通常是 (k, n)，我们需要把它变成 (min_dim, n)
    # 我们在下边增加 (min_dim - k) 行的 0
    Vt_final = np.zeros((min_dim, n), dtype=A.dtype)
    Vt_final[:k, :] = Vt_small
    
    return U_final, Sigma_final, Vt_final

def randomized_svd(
    data: Union[np.ndarray, torch.Tensor], 
    epsilon: float = 1e-2
) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
    """
    实现算法 5.1: 带有自动零填充(Zero-Padding)的随机化 SVD。
    
    功能:
    不管输入矩阵的真实秩是多少，输出的维度总是固定的 (Economy SVD 维度)。
    缺失的秩对应的奇异值设为 0，对应的向量设为 0。
    
    输入:
        data: (C, H, W) 或 (H, W)。例如 (3, 64, 64)。
        epsilon: 容差。
        
    输出 (假设输入 3, 64, 64):
        U:  (3, 64, 64)
        S:  (3, 64)      <-- 即使秩只有10，这里长度也是64，后54个为0
        Vh: (3, 64, 64)
    """
    
    # 1. 转换转 NumPy
    is_torch = False
    if isinstance(data, torch.Tensor):
        is_torch = True
        device = data.device
        dtype = data.dtype
        A_numpy = data.detach().cpu().numpy()
    else:
        A_numpy = data

    input_shape = A_numpy.shape
    
    # 2. 逐通道处理
    if len(input_shape) == 3:
        C, H, W = input_shape
        min_dim = min(H, W)
        
        # 预分配最终结果的内存 (直接按最大尺寸分配)
        U_batch = np.zeros((C, H, min_dim), dtype=A_numpy.dtype)
        S_batch = np.zeros((C, min_dim),    dtype=A_numpy.dtype)
        Vt_batch = np.zeros((C, min_dim, W), dtype=A_numpy.dtype)
        
        for i in range(C):
            # 处理单个通道
            u, s, vt = _randomized_svd_2d_padded(A_numpy[i], epsilon)
            
            # 填入 Batch 容器
            # 由于 _randomized_svd_2d_padded 保证返回固定维度，这里直接赋值即可
            U_batch[i] = u
            S_batch[i] = s
            Vt_batch[i] = vt
            
    elif len(input_shape) == 2:
        U_batch, S_batch, Vt_batch = _randomized_svd_2d_padded(A_numpy, epsilon)
        
    else:
        raise ValueError("仅支持 2D 或 3D 输入")

    # 3. 转回 Torch
    if is_torch:
        U_out = torch.from_numpy(U_batch).to(dtype=dtype, device=device)
        S_out = torch.from_numpy(S_batch).to(dtype=dtype, device=device)
        Vt_out = torch.from_numpy(Vt_batch).to(dtype=dtype, device=device)
        return U_out, S_out, Vt_out
    else:
        return U_batch, S_batch, Vt_batch

# --- 验证代码 ---
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 1. 构造一个低秩的输入 (3, 64, 64)
    # 我们故意让它的真实秩只有 10
    rank = 10
    U_true = torch.randn(3, 64, rank)
    S_true = torch.randn(3, rank)
    V_true = torch.randn(3, rank, 64)
    z_t = U_true @ torch.diag_embed(S_true) @ V_true
    
    print(f"输入张量形状: {z_t.shape}")
    print(f"真实设计的秩: {rank} (远小于 64)")
    
    # 2. 运行修改后的算法
    # 注意：epsilon 不要太小，否则它可能会试图去拟合噪声，算出一个很大的 k
    U_pad, S_pad, Vh_pad = randomized_svd(z_t, epsilon=1e-2)
    
    # 3. 验证维度 (必须是 64!)
    print("\n--- 维度检查 (期望全为 64) ---")
    print(f"U shape : {U_pad.shape}  -> {U_pad.shape == (3, 64, 64)}")
    print(f"S shape : {S_pad.shape}   -> {S_pad.shape == (3, 64)}")
    print(f"Vh shape: {Vh_pad.shape}  -> {Vh_pad.shape == (3, 64, 64)}")
    
    # 4. 验证补零效果
    print("\n--- 补零检查 ---")
    # 检查第 0 个通道，第 12 个奇异值 (应该远大于 rank=10，所以必须是 0)
    print(f"S[0, 11] (应为 0): {S_pad[0, 11]:.5f}")
    print(f"S[0, 8] (应为 0): {S_pad[0, 8]:.5f}")
    
    # 5. 验证重建精度
    recon = U_pad @ torch.diag_embed(S_pad) @ Vh_pad
    err = torch.norm(z_t - recon)
    print(f"\n重建误差: {err.item():.4f}")
    
    if S_pad.shape[1] == 64 and S_pad[0, -1] == 0:
        print("\n✅ 测试通过：维度符合 PyTorch 标准，且低秩部分已补零。")