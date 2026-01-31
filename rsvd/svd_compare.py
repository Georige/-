import numpy as np
from scipy import linalg
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# ==========================================
# 模块 1: 核心算法实现
# ==========================================

def randomized_svd(A: np.ndarray, n_components: int, oversample: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    随机化 SVD 分解。
    
    参数:
    A: 输入矩阵
    n_components: 目标秩 (我们希望提取多少个主要成分)
    oversample: 过采样参数，用于提高精度，通常 5-10 足够
    """
    M, N = A.shape
    
    # 随机投影矩阵 Omega
    # 维度说明: Omega 是 (N x (k + p))
    k = n_components
    p = oversample
    Omega = np.random.randn(N, k + p)
    
    # 1. 计算草图矩阵 (Sketch) Y = A @ Omega
    Y = A @ Omega
    
    # 2. QR 分解获取正交基 Q
    Q, _ = linalg.qr(Y, mode='economic')
    
    # 3. 投影原矩阵 B = Q.T @ A
    B = Q.T @ A
    
    # 4. 在小矩阵 B 上做标准 SVD
    U_tilde, S, Vt = linalg.svd(B, full_matrices=False)
    
    # 5. 还原 U = Q @ U_tilde
    U = Q @ U_tilde
    
    # 截断到目标秩 k
    return U[:, :k], S[:k], Vt[:k, :]

# ==========================================
# 模块 2: 统计型测试引擎
# ==========================================

def get_stats_for_rank(matrix_size: int, rank: int, n_repeats: int) -> Dict[str, float]:
    """
    针对特定的矩阵大小和特定的秩，重复运行 n_repeats 次实验，
    返回平均耗时。
    """
    svd_times = []
    rsvd_times = []
    
    for _ in range(n_repeats):
        # 每次都生成一个新的随机矩阵，避免缓存命中带来的偏差
        # 生成方法：A (size x size) = U (size x rank) @ V (rank x size)
        U_gen = np.random.randn(matrix_size, rank)
        V_gen = np.random.randn(rank, matrix_size)
        A = U_gen @ V_gen
        
        # --- 测试 Standard SVD ---
        t0 = time.perf_counter()
        linalg.svd(A, full_matrices=False)
        t1 = time.perf_counter()
        svd_times.append(t1 - t0)
        
        # --- 测试 Randomized SVD ---
        # 注意：我们告诉 rSVD 目标秩就是当前的 rank
        t0 = time.perf_counter()
        randomized_svd(A, n_components=rank)
        t1 = time.perf_counter()
        rsvd_times.append(t1 - t0)
        
    return {
        "svd_mean": np.mean(svd_times),
        "svd_std": np.std(svd_times),
        "rsvd_mean": np.mean(rsvd_times),
        "rsvd_std": np.std(rsvd_times)
    }

def run_comprehensive_benchmark(matrix_size: int, total_experiments: int):
    """
    运行综合基准测试。
    为了保证实验总数达到 total_experiments (例如 1500)，
    我们需要规划采样的秩点数 (steps) 和每个点的重复次数 (repeats)。
    """
    print(f"\n====== 开始大规模实验: Matrix {matrix_size}x{matrix_size} ======")
    
    # 策略：我们不测试每一个整数秩，而是采样大约 20-50 个不同的秩点
    # 这样每个点可以重复运行几十次，以获得准确的平均值
    
    num_rank_points = 30  # X轴上有30个数据点
    repeats_per_rank = total_experiments // num_rank_points # 每个点重复大概 50 次
    
    # 生成要测试的 rank 列表 (从 2 到 matrix_size，均匀分布)
    ranks_to_test = np.linspace(2, matrix_size, num_rank_points, dtype=int)
    # 确保秩不重复且有效
    ranks_to_test = np.unique(ranks_to_test)
    
    results = {
        "ranks": [],
        "svd_mean": [], "svd_std": [],
        "rsvd_mean": [], "rsvd_std": []
    }
    
    print(f"计划测试 {len(ranks_to_test)} 个不同的秩，每个秩重复 {repeats_per_rank} 次实验...")
    
    for r in ranks_to_test:
        stats = get_stats_for_rank(matrix_size, int(r), repeats_per_rank)
        
        results["ranks"].append(r)
        results["svd_mean"].append(stats["svd_mean"])
        results["svd_std"].append(stats["svd_std"])
        results["rsvd_mean"].append(stats["rsvd_mean"])
        results["rsvd_std"].append(stats["rsvd_std"])
        
        # 简单的进度条
        print(f"Rank {r:3d}: SVD Avg={stats['svd_mean']:.5f}s | rSVD Avg={stats['rsvd_mean']:.5f}s")

    return results

# ==========================================
# 模块 3: 专业级可视化
# ==========================================

def plot_benchmark_results(results: Dict, matrix_size: int):
    ranks = np.array(results["ranks"])
    svd_mean = np.array(results["svd_mean"])
    svd_std = np.array(results["svd_std"])
    rsvd_mean = np.array(results["rsvd_mean"])
    rsvd_std = np.array(results["rsvd_std"])
    
    plt.figure(figsize=(12, 7))
    
    # 绘制标准 SVD 曲线
    plt.plot(ranks, svd_mean, 'o-', label='Standard SVD (Mean)', color='blue', markersize=4)
    # 绘制误差带 (Mean ± 1 Std Dev)
    plt.fill_between(ranks, svd_mean - svd_std, svd_mean + svd_std, color='blue', alpha=0.15)
    
    # 绘制 rSVD 曲线
    plt.plot(ranks, rsvd_mean, 's-', label='Randomized SVD (Mean)', color='red', markersize=4)
    plt.fill_between(ranks, rsvd_mean - rsvd_std, rsvd_mean + rsvd_std, color='red', alpha=0.15)
    
    plt.title(f'Performance Comparison: SVD vs rSVD ({matrix_size}x{matrix_size})\n(Averaged over repeated trials)', fontsize=14)
    plt.xlabel('Matrix Rank', fontsize=12)
    plt.ylabel('Average Execution Time (seconds)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # 在图上标注加速比
    speedup = svd_mean[-1] / rsvd_mean[-1]
    plt.annotate(f'Max Rank Speedup: {speedup:.1f}x', 
                 xy=(ranks[-1], rsvd_mean[-1]), 
                 xytext=(ranks[-1] - (ranks[-1]*0.3), rsvd_mean[-1] + (svd_mean[-1]*0.1)),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.show()

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 预热 (Warm-up)
    print("正在进行系统预热...")
    dummy = np.random.randn(100, 100)
    linalg.svd(dummy)
    randomized_svd(dummy, 10)
    
    # 实验 1: 16x16 矩阵 (共1500次实验)
    # 小矩阵我们可能会发现 rSVD 并不占优势，因为 overhead 占比大
    results_16 = run_comprehensive_benchmark(matrix_size=16, total_experiments=1500)
    plot_benchmark_results(results_16, matrix_size=16)
    
    # 实验 2: 512x512 矩阵 (共1500次实验)
    # 这里我们应该能看到 rSVD 的显著优势
    results_512 = run_comprehensive_benchmark(matrix_size=512, total_experiments=1500)
    plot_benchmark_results(results_512, matrix_size=512)