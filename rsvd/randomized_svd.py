# 文件名: randomized_svd.py

import numpy as np
import time

# --- 模块化导入 ---
# 我们从 core_utils 模块中导入上一节课写的函数
# 确保 core_utils.py 和当前文件在同一目录下
try:
    from adaptive_range_finder import adaptive_randomized_range_finder
except ImportError:
    print("错误: 找不到 core_utils.py。请确保上一段代码已保存为该文件名。")
    exit(1)

def randomized_svd(A: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    实现算法 5.1: 直接随机化 SVD (Direct SVD)。
    
    利用 Range Finder 得到的 Q，将 A 分解为 U, S, Vt。
    A \approx U * diag(S) * Vt
    
    参数:
        A (np.ndarray): 输入矩阵 (m, n)。
        epsilon (float): 传递给 Range Finder 的容差。
        
    返回:
        U (np.ndarray): 左奇异向量 (m, k)。
        S (np.ndarray): 奇异值 (k,)。
        Vt (np.ndarray): 右奇异向量 (k, n)。
    """
    
    print("--- 第一阶段: 计算正交基 Q (Range Finder) ---")
    start_time = time.time()
    
    # 1. 使用我们在另一个文件中定义的算法计算 Q
    # Q 的形状是 (m, k)，其中 k << m
    Q = adaptive_randomized_range_finder(A, epsilon=epsilon)
    
    k = Q.shape[1]
    print(f"  >> 找到的秩 k = {k}")
    print(f"  >> Range Finder 耗时: {time.time() - start_time:.4f}s")
    
    print("--- 第二阶段: 降维与小矩阵分解 ---")
    # 2. 形成小矩阵 B
    # 理论公式: B = Q.T * A
    # Q.T 形状 (k, m), A 形状 (m, n) -> B 形状 (k, n)
    # 这一步将高维问题“压缩”到了低维空间
    B = Q.T @ A
    
    # 3. 对小矩阵 B 进行标准的确定性 SVD
    # 由于 B 只有 k 行 (k 通常很小)，这一步非常快
    # np.linalg.svd 默认返回的 full_matrices=True，我们需要设为 False (economy mode)
    S_hat, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    
    # S_hat 是 B 的左奇异向量，形状 (k, k)
    # Sigma 是奇异值，形状 (k,)
    # Vt 是右奇异向量，形状 (k, n) -> 这已经是我们要的最终 Vt 了！
    
    print("--- 第三阶段: 还原高维空间 ---")
    # 4. 计算最终的 U
    # 我们有 A ≈ Q * B = Q * (S_hat * Sigma * Vt)
    # 所以 U = Q * S_hat
    # Q (m, k) @ S_hat (k, k) -> U (m, k)
    U = Q @ S_hat
    
    return U, Sigma, Vt

# --- 系统集成测试 ---
if __name__ == "__main__":
    # 1. 生成测试数据 (与上次相同，保证可比性)
    np.random.seed(42)
    m, n = 2000, 200  # 稍微加大一点数据量
    true_rank = 15
    
    print(f"正在生成 ({m}x{n}) 的测试矩阵，真实秩为 {true_rank}...")
    U_true, _ = np.linalg.qr(np.random.normal(size=(m, true_rank)))
    V_true, _ = np.linalg.qr(np.random.normal(size=(n, true_rank)))
    S_true = np.sort(np.random.rand(true_rank))[::-1] * 10 # 奇异值
    A = U_true @ np.diag(S_true) @ V_true.T
    
    # 2. 执行随机化 SVD
    target_eps = 0.1
    print("\n开始执行 Randomized SVD...")
    U_approx, S_approx, Vt_approx = randomized_svd(A, epsilon=target_eps)
    
    # 3. 验证结果
    print(f"\n结果验证:")
    print(f"  >> 原始前5个奇异值: {S_true[:5]}")
    print(f"  >> 计算前5个奇异值: {S_approx[:5]}")
    
    # 验证重建误差 || A - U S Vt ||
    A_reconstructed = U_approx @ np.diag(S_approx) @ Vt_approx
    error = np.linalg.norm(A - A_reconstructed, ord=2)
    print(f"  >> 谱范数误差: {error:.6f}")
    
    if error < target_eps:
        print("\n✅ 系统测试通过：重建精度符合要求。")
    else:
        print("\n⚠️ 系统测试警告：误差可能偏大。")