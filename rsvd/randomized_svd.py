import numpy as np
import time
from typing import Tuple, Union, List

# --- 模块化导入 ---
try:
    from adaptive_range_finder import adaptive_randomized_range_finder
except ImportError:
    print("错误: 找不到 core_utils.py。请确保上一段代码已保存为该文件名。")
    exit(1)

def randomized_svd(
    data: np.ndarray, 
    epsilon: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    实现算法 5.1: 直接随机化 SVD (支持 2D 矩阵和 3D 张量)。
    
    如果是 3D 张量 (C, H, W)，会自动展平为 (C*H, W) 进行计算，
    并在返回时尝试还原 U 的维度。
    
    参数:
        data (np.ndarray): 输入数据，形状可以是 (m, n) 或 (c, h, w)。
        epsilon (float): 容差阈值。
        
    返回:
        U (np.ndarray): 左奇异向量。
        S (np.ndarray): 奇异值。
        Vt (np.ndarray): 右奇异向量。
    """
    
    # --- 阶段 0: 数据预处理 (张量适配) ---
    original_shape = data.shape
    is_3d = False
    
    if len(original_shape) == 3:
        # 处理 3D 张量: (Channels, Height, Width) -> (3, 64, 64)
        print(f"检测到 3D 输入 {original_shape}，正在执行模式转换 (Unfolding)...")
        is_3d = True
        c, h, w = original_shape
        
        # 策略: 将 (C, H, W) 展平为 (C*H, W)
        # 也就是把每个通道的每一行都堆叠起来，形成一个“高瘦”的矩阵
        # 形状变为 (192, 64)
        A = data.reshape(c * h, w)
        print(f"  >> 转换后矩阵形状: {A.shape}")
    elif len(original_shape) == 2:
        # 标准 2D 矩阵
        A = data
    else:
        raise ValueError(f"不支持的维度: {original_shape}。仅支持 2D 或 3D 输入。")

    print("--- 第一阶段: 计算正交基 Q (Range Finder) ---")
    start_time = time.time()
    
    # 1. 计算 Q
    Q = adaptive_randomized_range_finder(A, epsilon=epsilon)
    
    k = Q.shape[1]
    print(f"  >> 找到的秩 k = {k}")
    print(f"  >> Range Finder 耗时: {time.time() - start_time:.4f}s")
    
    print("--- 第二阶段: 降维与小矩阵分解 ---")
    # 2. 形成小矩阵 B = Q.T @ A
    B = Q.T @ A
    
    # 3. 对小矩阵 B 进行标准 SVD (Economy mode)
    # S_hat: (k, k), Sigma: (k,), Vt: (k, n)
    S_hat, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    
    print("--- 第三阶段: 还原高维空间 ---")
    # 4. 计算最终的 U = Q @ S_hat
    # U 的形状目前是 (m, k)，即 (C*H, k)
    U = Q @ S_hat
    
    # --- 阶段 4: 结果后处理 (维度还原) ---
    if is_3d:
        # 如果输入是 (C, H, W)，我们将 U 从 (C*H, k) 还原为 (C, H, k)
        # 这样你可以保留空间结构信息
        # 注意: Vt 通常保持 (k, W)，代表特征在宽度维度的分布
        try:
            c, h, w = original_shape
            U_reshaped = U.reshape(c, h, k)
            print(f"  >> 已将 U 还原为 3D 结构: {U_reshaped.shape}")
            return U_reshaped, Sigma, Vt
        except Exception as e:
            print(f"  >> 警告: U 维度还原失败 ({e})，返回 2D 形式。")
            return U, Sigma, Vt
            
    return U, Sigma, Vt

# --- 系统集成测试 (针对 3D 图片数据) ---
if __name__ == "__main__":
    np.random.seed(42)
    
    # 1. 模拟一张 (3, 64, 64) 的图片
    # 我们故意制造一些低秩结构：背景(平滑) + 前景(物体)
    channels, height, width = 3, 64, 64
    print(f"\n正在生成模拟图片数据 ({channels}, {height}, {width})...")
    
    # 创建基础模式
    base_pattern = np.zeros((height, width))
    for i in range(height):
        base_pattern[i, :] = np.sin(i / 5.0)  # 简单的波纹
        
    img_tensor = np.zeros((channels, height, width))
    for c in range(channels):
        # 每个通道有些许偏移
        img_tensor[c, :, :] = base_pattern * (c + 1) + np.random.normal(0, 0.1, (height, width))
        
    # 2. 执行随机化 SVD
    target_eps = 0.5 # 对于图片，容差可以稍微大一点
    print("\n开始执行 Randomized SVD...")
    
    # 这一步会自动处理 reshape
    U_approx, S_approx, Vt_approx = randomized_svd(img_tensor, epsilon=target_eps)
    
    # 3. 验证结果形状
    print(f"\n结果分析:")
    print(f"  >> U shape: {U_approx.shape} (预期: 3, 64, k)")
    print(f"  >> S shape: {S_approx.shape} (预期: k)")
    print(f"  >> Vt shape: {Vt_approx.shape} (预期: k, 64)")
    
    # 4. 尝试重建图片验证精度
    # 重建公式: A ≈ U * S * Vt
    # 需要注意维度的对齐: 
    # U(3, 64, k) dot diag(S) dot Vt(k, 64)
    # 为了计算方便，先把 U 变回 2D: (192, k)
    k = len(S_approx)
    U_2d = U_approx.reshape(channels * height, k)
    
    # 重建 2D 矩阵
    img_reconstructed_2d = U_2d @ np.diag(S_approx) @ Vt_approx
    
    # 变回 3D
    img_reconstructed = img_reconstructed_2d.reshape(channels, height, width)
    
    error = np.linalg.norm(img_tensor - img_reconstructed)
    print(f"  >> 重建误差 (Frobenius Norm): {error:.4f}")
    
    if error < 10.0: # 图片数据的数值通常较大，误差绝对值会比之前大
        print("\n✅ 3D Tensor 测试通过。")
    else:
        print("\n⚠️ 误差较大，请检查参数。")