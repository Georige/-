import numpy as np
from scipy import linalg
import time
import matplotlib.pyplot as plt
from typing import Tuple, List

# ==========================================
# 模块 1: 算法实现
# ==========================================

def randomized_svd(A: np.ndarray, epsilon: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用随机化方法计算矩阵的SVD分解。
    
    参数:
    A (np.ndarray): 输入矩阵 (M x N)
    epsilon (float): 用于确定目标秩的冗余量 (这里简化实现，将其作为过采样参数)
    
    返回:
    U, S, Vt: SVD 分解的结果
    """
    M, N = A.shape
    
    # 1. 确定目标秩 k。在实际应用中，k 应该比矩阵的有效秩略大。
    # 为了简化，我们假设我们想提取前 k 个主成分。这里动态设置为 min(M, N) 的一部分。
    k = max(1, int(min(M, N) * 0.5)) # 假设截断到 50% 的秩
    
    # 过采样参数，增加随机矩阵的列数以提高精度
    p = 5 
    
    # 2. 生成随机高斯矩阵 Omega (N x (k+p))
    Omega = np.random.randn(N, k + p)
    
    # 3. 构造子空间 Y = A * Omega (M x (k+p))
    Y = A @ Omega
    
    # 4. 对 Y 进行 QR 分解，得到正交基 Q
    Q, _ = linalg.qr(Y, mode='economic')
    
    # 5. 投影到子空间: B = Q.T * A (k+p x N)
    B = Q.T @ A
    
    # 6. 对小矩阵 B 进行标准 SVD
    U_tilde, S, Vt = linalg.svd(B, full_matrices=False)
    
    # 7. 恢复 U = Q * U_tilde
    U = Q @ U_tilde
    
    # 根据 k 截断返回
    return U[:, :k], S[:k], Vt[:k, :]


# ==========================================
# 模块 2: 数据生成与测试引擎
# ==========================================

def generate_rank_k_matrix(m: int, n: int, rank: int) -> np.ndarray:
    """
    生成一个大小为 m x n 且精确秩为 rank 的矩阵。
    """
    # 通过两个随机矩阵相乘生成：(m x rank) @ (rank x n)
    U = np.random.randn(m, rank)
    V = np.random.randn(rank, n)
    return U @ V

def run_benchmark(matrix_size: int, num_trials: int = 1500):
    """
    针对特定尺寸的矩阵运行基准测试，秩随试验次数线性增加。
    """
    print(f"\n--- 开始实验: 矩阵大小 {matrix_size}x{matrix_size}, 试验次数 {num_trials} ---")
    
    svd_times = []
    rsvd_times = []
    ranks = []
    
    total_svd_time = 0.0
    total_rsvd_time = 0.0

    # 预热 (Warm-up): 避免 Python 的首次加载开销影响时间统计
    dummy = np.random.randn(matrix_size, matrix_size)
    linalg.svd(dummy)
    randomized_svd(dummy)

    for i in range(num_trials):
        # 秩随着迭代次数从 1 线性增加到 max_rank
        current_rank = max(1, int((i / (num_trials - 1)) * matrix_size))
        ranks.append(current_rank)
        
        A = generate_rank_k_matrix(matrix_size, matrix_size, current_rank)
        
        # 测试标准 SVD
        start_time = time.perf_counter()
        linalg.svd(A, full_matrices=False)
        svd_time = time.perf_counter() - start_time
        svd_times.append(svd_time)
        total_svd_time += svd_time
        
        # 测试 rSVD
        start_time = time.perf_counter()
        randomized_svd(A)
        rsvd_time = time.perf_counter() - start_time
        rsvd_times.append(rsvd_time)
        total_rsvd_time += rsvd_time

        if (i + 1) % 300 == 0:
            print(f"  已处理 {i + 1} 个矩阵...")

    print(f"总计耗时 (Standard SVD): {total_svd_time:.4f} 秒")
    print(f"总计耗时 (Randomized SVD): {total_rsvd_time:.4f} 秒")
    
    return ranks, svd_times, rsvd_times

# ==========================================
# 模块 3: 可视化
# ==========================================

def plot_results(ranks: List[int], svd_times: List[float], rsvd_times: List[float], matrix_size: int):
    """
    绘制随秩增加，耗时变化的图表。
    这里我们使用移动平均法(Moving Average)平滑曲线，以便观察趋势。
    """
    window_size = 50 # 平滑窗口
    svd_smooth = np.convolve(svd_times, np.ones(window_size)/window_size, mode='valid')
    rsvd_smooth = np.convolve(rsvd_times, np.ones(window_size)/window_size, mode='valid')
    ranks_smooth = ranks[:len(svd_smooth)]

    plt.figure(figsize=(10, 6))
    plt.plot(ranks_smooth, svd_smooth, label='Standard SVD', color='blue', alpha=0.7)
    plt.plot(ranks_smooth, rsvd_smooth, label='Randomized SVD', color='red', alpha=0.7)
    
    plt.title(f'SVD vs Randomized SVD Execution Time ({matrix_size}x{matrix_size})')
    plt.xlabel('Matrix Rank')
    plt.ylabel('Execution Time (seconds) - Smoothed')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================
# 主程序执行入口
# ==========================================
if __name__ == "__main__":
    # 实验 1: 16x16 小规模矩阵
    ranks_16, svd_16, rsvd_16 = run_benchmark(matrix_size=16, num_trials=1500)
    plot_results(ranks_16, svd_16, rsvd_16, matrix_size=16)

    # 实验 2: 512x512 大规模矩阵
    ranks_512, svd_512, rsvd_512 = run_benchmark(matrix_size=512, num_trials=1500)
    plot_results(ranks_512, svd_512, rsvd_512, matrix_size=512)