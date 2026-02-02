import numpy as np
from scipy.linalg import svd

def svd_hard_thresholding(X, k):
    """
    执行矩阵的奇异值硬阈值化 (Truncated SVD)
    保留前 k 个最大的奇异值，其余置为 0
    """
    U, s, Vh = svd(X, full_matrices=False)
    
    # 只保留前 k 个奇异值
    s_k = np.zeros_like(s)
    s_k[:k] = s[:k]
    
    # 重构矩阵
    return np.dot(U, np.dot(np.diag(s_k), Vh))

def fast_grahtp_matrix(f_grad, X0, k, eta, max_iter=100, tol=1e-6):
    """
    复现图中 'Fast GraHTP' 的矩阵秩约束版本
    
    参数:
    f_grad: 函数，输入矩阵 X，返回其梯度矩阵 grad(f(X))
    X0: 初始矩阵
    k: 目标秩 (SVD rank constraint)
    eta: 步长 (Step size)
    max_iter: 最大迭代次数
    tol: 停止收敛的阈值
    """
    X_t = np.copy(X0)
    
    for t in range(max_iter):
        X_prev = np.copy(X_t)
        
        # (S1) 梯度下降步骤: x_tilde = x - eta * grad(f(x))
        grad = f_grad(X_t)
        X_tilde = X_t - eta * grad
        
        # (S2) 矩阵硬阈值化步骤: 保留 top-k 奇异值 (等同于矩阵版本的截断)
        X_t = svd_hard_thresholding(X_tilde, k)
        
        # 检查收敛 (Halting condition)
        diff = np.linalg.norm(X_t - X_prev, 'fro')
        if diff < tol:
            print(f"Converged at iteration {t}")
            break
            
    return X_t

# --- 使用示例 (矩阵补全/恢复) ---

# 1. 构造一个低秩矩阵 (Rank 2)
m, n = 50, 50
rank_true = 2
U_true = np.random.randn(m, rank_true)
V_true = np.random.randn(rank_true, n)
X_true = U_true @ V_true

# 2. 定义目标函数梯度 (以最小二乘为例: f(X) = 0.5 * ||X - X_true||^2)
# 实际场景中可能只知道部分观测值
def simple_grad(X):
    return X - X_true

# 3. 运行算法
X_init = np.zeros((m, n))
k_constraint = 4  # 假设我们知道秩是 2
learning_rate = 0.8

result = fast_grahtp_matrix(simple_grad, X_init, k_constraint, learning_rate)

print(f"恢复矩阵与原矩阵的 Frobenius 范数距离: {np.linalg.norm(result - X_true, 'fro'):.2e}")