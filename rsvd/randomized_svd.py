import numpy as np
import torch
from typing import Tuple, Union

# --- 模块化导入 ---
from adaptive_range_finder import adaptive_randomized_range_finder

def _randomized_svd_2d_padded(
    A: Union[np.ndarray, torch.Tensor], 
    epsilon: float
) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
    """
    内部函数：执行自适应 SVD，并支持 PyTorch/NumPy 自动切换和零填充。
    """
    # 1. 检测环境
    is_torch = isinstance(A, torch.Tensor)
    
    # 获取形状
    m, n = A.shape
    min_dim = min(m, n) 
    
    # 2. 自适应计算 Range (这一步会调用我们刚修好的 adaptive_range_finder)
    # Q 的类型会和 A 保持一致 (GPU Tensor 或 NumPy)
    Q = adaptive_randomized_range_finder(A, epsilon=epsilon)
    
    # 计算 B = Q.T * A
    # PyTorch 和 NumPy 都支持 @ 运算符
    B = Q.T @ A
    
    # 3. 对小矩阵 B 进行标准 SVD (区分框架)
    if is_torch:
        # PyTorch 路径
        # S_hat: (k, k), Sigma: (k,), Vt: (k, n)
        # 注意：torch.linalg.svd 返回的 U 对应这里的 S_hat
        S_hat, Sigma_small, Vt_small = torch.linalg.svd(B, full_matrices=False)
        
        # 还原 U_small = Q @ S_hat
        U_small = Q @ S_hat
        
        # 获取当前秩 k
        k = Sigma_small.shape[0]
        
        # --- Padding (PyTorch) ---
        if k < min_dim:
            # 补全 S
            Sigma_final = torch.zeros(min_dim, dtype=A.dtype, device=A.device)
            Sigma_final[:k] = Sigma_small
            
            # 补全 U
            U_final = torch.zeros((m, min_dim), dtype=A.dtype, device=A.device)
            U_final[:, :k] = U_small
            
            # 补全 Vt
            Vt_final = torch.zeros((min_dim, n), dtype=A.dtype, device=A.device)
            Vt_final[:k, :] = Vt_small
            
            return U_final, Sigma_final, Vt_final
        else:
            # 截断（防止 k > min_dim 的浮点误差情况）
            return U_small[:, :min_dim], Sigma_small[:min_dim], Vt_small[:min_dim, :]
            
    else:
        # NumPy 路径 (保持原有逻辑)
        S_hat, Sigma_small, Vt_small = np.linalg.svd(B, full_matrices=False)
        U_small = Q @ S_hat
        k = Sigma_small.shape[0]
        
        if k < min_dim:
            Sigma_final = np.zeros(min_dim, dtype=A.dtype)
            Sigma_final[:k] = Sigma_small
            
            U_final = np.zeros((m, min_dim), dtype=A.dtype)
            U_final[:, :k] = U_small
            
            Vt_final = np.zeros((min_dim, n), dtype=A.dtype)
            Vt_final[:k, :] = Vt_small
            return U_final, Sigma_final, Vt_final
        else:
            return U_small[:, :min_dim], Sigma_small[:min_dim], Vt_small[:min_dim, :]

def randomized_svd(
    data: Union[np.ndarray, torch.Tensor], 
    epsilon: float = 1e-2
) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
    """
    实现算法 5.1: 逐通道随机化 SVD (支持 Batch/Channel-wise)。
    完全兼容 PyTorch GPU Tensor 流水线，无需 CPU 转换。
    
    输出维度 (假设输入 3, 64, 64):
        U:  (3, 64, 64)
        S:  (3, 64)      (零填充对齐)
        Vh: (3, 64, 64)
    """
    
    # 1. 基础信息获取
    is_torch = isinstance(data, torch.Tensor)
    input_shape = data.shape
    
    # 2. 逐通道处理逻辑
    if len(input_shape) == 3:
        # (C, H, W) 模式
        C, H, W = input_shape
        min_dim = min(H, W)
        
        # 准备容器
        if is_torch:
            # 直接在 GPU 上分配内存
            U_batch = torch.zeros((C, H, min_dim), dtype=data.dtype, device=data.device)
            S_batch = torch.zeros((C, min_dim),    dtype=data.dtype, device=data.device)
            Vt_batch = torch.zeros((C, min_dim, W), dtype=data.dtype, device=data.device)
        else:
            U_batch = np.zeros((C, H, min_dim), dtype=data.dtype)
            S_batch = np.zeros((C, min_dim),    dtype=data.dtype)
            Vt_batch = np.zeros((C, min_dim, W), dtype=data.dtype)
        
        for i in range(C):
            # 取出单个通道 (保持 Tensor 属性)
            # data[i] 依然是 GPU tensor
            u, s, vt = _randomized_svd_2d_padded(data[i], epsilon)
            
            U_batch[i] = u
            S_batch[i] = s
            Vt_batch[i] = vt
            
        return U_batch, S_batch, Vt_batch
            
    elif len(input_shape) == 2:
        # 2D 模式直接调用
        return _randomized_svd_2d_padded(data, epsilon)
        
    else:
        raise ValueError(f"仅支持 2D 或 3D 输入，当前形状: {input_shape}")

# --- 验证代码 (确保 GPU 流程通畅) ---
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("正在测试 CUDA GPU 模式...")
        device = "cuda:0"
        
        # 1. 创建 GPU 数据 (3, 64, 64)
        # 模拟真实秩 rank=10
        rank = 10
        U = torch.randn(3, 64, rank, device=device)
        S = torch.randn(3, rank, device=device)
        V = torch.randn(3, rank, 64, device=device)
        z_t = U @ torch.diag_embed(S) @ V
        
        print(f"输入数据位于: {z_t.device}")
        
        # 2. 运行算法
        # 期望：没有任何报错，且输出依然在 GPU 上
        U_out, S_out, Vh_out = randomized_svd(z_t, epsilon=1e-2)
        
        print(f"输出 U 位于: {U_out.device}")
        print(f"输出形状: {U_out.shape}, {S_out.shape}, {Vh_out.shape}")
        
        if U_out.is_cuda:
            print("✅ 测试通过：全链路 GPU 计算成功！")
        else:
            print("❌ 测试失败：数据回落到了 CPU。")
    else:
        print("未检测到 GPU，跳过 GPU 测试。")