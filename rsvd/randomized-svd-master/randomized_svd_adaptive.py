import time
import warnings
from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy import linalg
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
# from sklearn.utils._array_api import (
#     _average,
#     _is_numpy_namespace,
#     _max_precision_float_dtype,
#     _nanmean,
#     _nansum,
#     device,
#     get_namespace,
#     get_namespace_and_device,
# )


def adaptive_randomized_range_finder(A, tol, n_iter=2, block_size=10, random_state=None):
    """
    自适应 Range Finder (算法 4.2)
    动态构建 Q，直到近似误差小于 tol。
    """
    rng = check_random_state(random_state)
    m, n = A.shape
    
    # 1. 初始化
    Q = np.zeros((m, 0))
    # 初始的一组随机向量
    Omega = rng.normal(size=(n, block_size))
    Y = A @ Omega
    
    # 进行幂迭代以提高精度 (Power Iteration)
    # Y = (A A.T)^q A Omega
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
        
    j = 0
    # 容差归一化：论文建议除以一个系数，或者直接用绝对误差
    # limit = tol / (10 * np.sqrt(2 / np.pi)) 
    # 为了工程简单，这里直接比较谱范数代理
    
    while True:
        # 2. 检查当前块的误差
        # 如果是第一轮，没有上一轮的 Q，不需要投影
        # 如果不是第一轮，减去已经在 Q 中的分量
        if j > 0:
            # Gram-Schmidt 正交化 / 投影
            # Y_block = (I - Q Q.T) Y_block
            # 注意：这里的 Q 是累积的
            Y[:, j:] = Y[:, j:] - Q @ (Q.T @ Y[:, j:])
        
        # 3. 检查停止条件
        # 计算当前块中最大的列范数，作为误差的估计
        current_block_norms = np.linalg.norm(Y[:, j:], axis=0)
        max_norm = np.max(current_block_norms)
        
        if max_norm < tol:
            break
            
        # 安全守卫：防止死循环（比如 rank > min(m, n)）
        if Q.shape[1] >= min(m, n):
            break

        # 4. 更新 Q
        # 对当前块 Y[:, j : j+block_size] 进行 QR 分解得到新的基
        # 简单起见，我们逐列处理或使用 QR
        Q_new_block, _ = linalg.qr(Y[:, j:], mode='economic')
        
        # 将新基加入 Q
        Q = np.column_stack((Q, Q_new_block))
        
        # 5. 准备下一轮
        # 生成新的测试向量
        Omega_new = rng.normal(size=(n, block_size))
        Y_new = A @ Omega_new
        
        # 幂迭代
        for _ in range(n_iter):
            Y_new = A @ (A.T @ Y_new)
            
        # 对旧的 Q 正交化 (Double Orthogonalization for stability)
        Y_new = Y_new - Q @ (Q.T @ Y_new)
        
        # 拼接到 Y 中以便下一轮处理
        Y = np.column_stack((Y, Y_new))
        
        # 更新索引
        j += block_size
        
    return Q

def randomized_svd_adaptive(
    M,
    tol=1e-2,  # <--- 变化点：用 tol 替代 n_components
    *,
    n_iter="auto",
    transpose="auto",
    flip_sign=True,
    random_state=None,
    block_size=10 # 新增参数：每次探索的步长
):
    """
    未知秩的随机化 SVD 主函数。
    """
    # ... (省略 sparse 检查，与原版一致) ...
    #xp, is_array_api_compliant = get_namespace(M)

    if sparse.issparse(M) and M.format in ("lil", "dok"):
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(M).__name__),
            sparse.SparseEfficiencyWarning,
        )

    random_state = check_random_state(random_state)
    n_samples, n_features = M.shape

    # 自动设置迭代次数 (依然保留这个逻辑)
    if n_iter == "auto":
        n_iter = 4 # 默认值

    # 转置逻辑 (保持不变，确保我们在短边采样)
    if transpose == "auto":
        transpose = n_samples < n_features
    if transpose:
        M = M.T

    # --- 变化点：调用自适应 Range Finder ---
    # 我们不再传入 size=n_random，而是传入 tol
    Q = adaptive_randomized_range_finder(
        M,
        tol=tol,
        n_iter=n_iter,
        block_size=block_size,
        random_state=random_state,
    )
    
    # 获取自适应找到的秩 k
    k_found = Q.shape[1]
    # print(f"自适应算法找到的有效秩: {k_found}")

    # --- 后续逻辑与原版基本一致 ---
    
    # 1. 投影: B = Q.T @ M
    # B 的形状是 (k_found, n)
    B = Q.T @ M

    # 2. 小矩阵 SVD
    Uhat, s, Vt = linalg.svd(B, full_matrices=False)
    
    # 3. 还原 U
    U = Q @ Uhat

    # 4. 符号修正
    if flip_sign:
        if not transpose:
            U, Vt = svd_flip(U, Vt)
        else:
            U, Vt = svd_flip(U, Vt, u_based_decision=False)

    # 5. 返回结果 (不需要截断，因为本来就是算多少返回多少)
    if transpose:
        return Vt.T, s, U.T
    else:
        return U, s, Vt
    


# ==========================================
# 2. 测试数据生成器
# ==========================================

def generate_data(m=2000, n=2000, effective_rank=50, noise_level=0.1):
    """生成一个低秩矩阵 + 噪声"""
    np.random.seed(42)
    
    # 构造奇异值：前 50 个很大，后面衰减很快
    s_true = np.zeros(min(m, n))
    s_true[:effective_rank] = np.linspace(100, 10, effective_rank) # 信号部分
    s_true[effective_rank:] = np.linspace(noise_level, 0, min(m, n) - effective_rank) # 噪声部分
    
    # 随机生成正交矩阵 U 和 V
    # 为了速度，这里简化生成过程
    U_true, _ = linalg.qr(np.random.normal(size=(m, effective_rank + 20)), mode='economic')
    V_true, _ = linalg.qr(np.random.normal(size=(n, effective_rank + 20)), mode='economic')
    
    # 稍微截断以匹配维度
    k_gen = min(U_true.shape[1], len(s_true))
    A = U_true[:, :k_gen] @ np.diag(s_true[:k_gen]) @ V_true[:, :k_gen].T
    
    return A, s_true

# ==========================================
# 3. 运行基准测试
# ==========================================

if __name__ == "__main__":
    # 设置参数
    M_SIZE = 2000
    EFFECTIVE_RANK = 50
    NOISE_LEVEL = 0.5
    
    print(f"正在生成 {M_SIZE}x{M_SIZE} 的测试矩阵 (有效秩 approx {EFFECTIVE_RANK})...")
    A, S_true = generate_data(m=M_SIZE, n=M_SIZE, effective_rank=EFFECTIVE_RANK, noise_level=NOISE_LEVEL)
    
    # 定义不同的容差等级
    # 10.0: 非常粗糙，可能连主要特征都抓不全
    # 1.0:  应该能抓到大部分强信号
    # 0.1:  应该能完美抓到有效秩，并开始触及噪声
    # 0.01: 精度非常高，会保留很多噪声维度
    tolerances = [10.0, 5.0, 1.0, 0.5, 0.1, 0.01]
    
    results_time = []
    results_rank = []
    results_error = []
    
    print("\n{:<10} | {:<10} | {:<10} | {:<15}".format("Tol", "Time(s)", "Rank(k)", "Error(Norm)"))
    print("-" * 55)
    
    for tol in tolerances:
        start_time = time.time()
        
        # --- 运行算法 ---
        U_est, S_est, Vt_est = randomized_svd_adaptive(A, tol=tol, n_iter=1, block_size=10)
        # ----------------
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # 计算找到的秩
        k_found = len(S_est)
        
        # 计算近似误差 (谱范数/F范数太慢，这里用重建误差估计)
        # Error ≈ || A - U S Vt ||
        # 为了速度，只随机抽样一部分点或者计算 Frobenius 范数
        # 严格计算：diff = A - U_est @ np.diag(S_est) @ Vt_est
        # error = linalg.norm(diff)
        
        # 快速估算误差：对比第 k+1 个真实奇异值 (理论误差下界)
        # 或者直接看 S_est 的最后一个值是否接近 tol
        last_sigma = S_est[-1] if len(S_est) > 0 else 0
        
        results_time.append(elapsed)
        results_rank.append(k_found)
        results_error.append(last_sigma)
        
        print(f"{tol:<10} | {elapsed:<10.4f} | {k_found:<10} | {last_sigma:<15.4f}")

    # ==========================================
    # 4. 绘图分析
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Tolerance (log scale)')
    ax1.set_ylabel('Execution Time (s)', color=color)
    ax1.plot(tolerances, results_time, color=color, marker='o', linestyle='-', label='Time')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.invert_xaxis() # 让 x 轴从大到小 (容差越小越靠右)

    ax2 = ax1.twinx()  # 共享 x 轴
    color = 'tab:blue'
    ax2.set_ylabel('Found Rank (k)', color=color)
    ax2.plot(tolerances, results_rank, color=color, marker='s', linestyle='--', label='Rank')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 标出真实秩的参考线
    plt.axhline(y=EFFECTIVE_RANK, color='green', linestyle=':', label='True Effective Rank')

    plt.title(f'Adaptive SVD Efficiency: Tolerance vs Time/Rank\n(Matrix: {M_SIZE}x{M_SIZE}, True Rank: {EFFECTIVE_RANK})')
    fig.tight_layout()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 保存或显示
    print("\n正在生成图表...")
    plt.show()
    