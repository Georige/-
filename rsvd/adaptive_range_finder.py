import numpy as np
import torch
from typing import Union, List

def adaptive_randomized_range_finder(
    A: Union[np.ndarray, torch.Tensor], 
    epsilon: float, 
    r: int = 10
) -> Union[np.ndarray, torch.Tensor]:
    """
    实现算法 4.2: 自适应随机化 Range Finder (PyTorch/NumPy 通用版)。
    
    该函数计算矩阵 A 的正交基 Q，使得近似误差在概率上小于 epsilon。
    自动适配 CPU(NumPy) 或 GPU(PyTorch)。
    """
    
    # --- 1. 环境检测与适配 ---
    is_torch = False
    device = None
    dtype = None
    
    if isinstance(A, torch.Tensor):
        is_torch = True
        device = A.device
        dtype = A.dtype
        # 获取维度
        m, n = A.shape
    else:
        # NumPy 模式
        m, n = A.shape
        dtype = A.dtype

    # --- 2. 辅助函数 (屏蔽框架差异) ---
    def make_random(shape):
        if is_torch:
            return torch.randn(shape, device=device, dtype=dtype)
        else:
            return np.random.normal(size=shape).astype(dtype)
            
    def calc_norm(vec):
        if is_torch:
            return torch.norm(vec)
        else:
            return np.linalg.norm(vec)
            
    def calc_dot(v1, v2):
        if is_torch:
            return torch.dot(v1, v2)
        else:
            return np.dot(v1, v2)
            
    def mat_mul_vec(mat, vec):
        # 矩阵乘向量
        return mat @ vec

    # --- 步骤 1: 初始化 ---
    # Draw standard Gaussian vectors omega^(1)...omega^(r)
    Omega = make_random((n, r))
    
    # --- 步骤 2: 初始采样 ---
    # Compute Y = A * Omega
    # 注意：为了保持动态特性，我们用列表存储向量
    Y = []
    for i in range(r):
        # 取出第 i 列
        omega_col = Omega[:, i]
        y_col = mat_mul_vec(A, omega_col)
        Y.append(y_col)
    
    # --- 步骤 3 & 4: 初始化循环变量 ---
    j = 0
    Q = []  # 存放正交基向量
    
    # 计算阈值 limit
    # np.sqrt(2 / np.pi) 约等于 0.798
    const_factor = 0.79788456
    limit = epsilon / (10 * const_factor)
    
    # --- 步骤 5: While 循环 ---
    # 只要前瞻窗口内的向量能量还很大，就继续寻找
    while True:
        # 检查是否越界 (防止极其罕见的无限循环)
        if j >= n: 
            break
            
        # 获取当前窗口内的向量 Y[j : j+r]
        # 如果窗口超出了 Y 的当前长度，说明需要生成新的 (虽然后面的逻辑会生成，但这里做个防守)
        current_window = Y[j : j+r]
        if not current_window:
            break
            
        # 计算窗口内每个向量的范数
        norms = [calc_norm(y).item() for y in current_window] # .item() 转为 python float 比较
        max_norm = max(norms)
        
        # 停止条件
        if max_norm <= limit:
            break
            
        # --- 步骤 7: 投影 (Gram-Schmidt) ---
        # 这里的 Y[j] 实际上已经被之前的 Q 正交化过了(在步骤13)，
        # 但为了数值稳定性，或者如果是第一轮，我们需要确保它正交。
        y_current = Y[j]
        
        # Double Orthogonalization (数值稳定性关键)
        for _ in range(2): # 做两次以防万一，通常一次也够
            for q_prev in Q:
                projection = calc_dot(q_prev, y_current)
                y_current = y_current - q_prev * projection
        
        # --- 步骤 8: 归一化 ---
        norm_y = calc_norm(y_current)
        
        if norm_y < 1e-15:
            # 线性相关，跳过
            j += 1
            continue
            
        q_new = y_current / norm_y
        Q.append(q_new)
        
        # --- 步骤 10: 生成新的高斯向量 ---
        omega_new = make_random((n,))
        
        # --- 步骤 11: 计算新样本 ---
        # y_new = (I - Q Q*) A omega_new
        # 先算 A * omega
        y_new = mat_mul_vec(A, omega_new)
        
        # 立即对现有的 Q 进行正交化
        for q in Q:
            y_new = y_new - q * calc_dot(q, y_new)
            
        Y.append(y_new)
        
        # --- 步骤 12 & 13: 更新前瞻窗口内的向量 ---
        # Y[i] = Y[i] - q_new * <q_new, Y[i]>
        # 范围: j+1 到 j+r (注意 Python切片是左闭右开，但这里不仅是切片，是由于 append 导致 len 增加)
        # 我们只需要更新目前列表中位于 j 之后的所有向量
        for i in range(j + 1, len(Y)):
            proj = calc_dot(q_new, Y[i])
            Y[i] = Y[i] - q_new * proj
            
        j += 1

    # --- 步骤 16: 构建最终矩阵 ---
    if not Q:
        # 返回空矩阵
        if is_torch:
            return torch.zeros((m, 0), device=device, dtype=dtype)
        else:
            return np.zeros((m, 0), dtype=dtype)
    
    # 堆叠结果
    if is_torch:
        Q_matrix = torch.stack(Q, dim=1)
    else:
        Q_matrix = np.column_stack(Q)
        
    return Q_matrix
# --- 单元测试/用法示例 ---
if __name__ == "__main__":
    # 1. 创建一个具有特定秩的合成矩阵来测试
    # 假设 m=1000, n=100, 真实秩=10
    np.random.seed(42) # 固定随机种子以复现结果
    m, n = 1000, 100
    true_rank = 10
    
    # 构造低秩矩阵 A = U * S * V.T
    U_true, _ = np.linalg.qr(np.random.normal(size=(m, true_rank)))
    V_true, _ = np.linalg.qr(np.random.normal(size=(n, true_rank)))
    S_true = np.diag(np.linspace(10, 1, true_rank)) # 奇异值从 10 降到 1
    A = U_true @ S_true @ V_true.T
    
    print(f"原始矩阵形状: {A.shape}, 真实秩: {true_rank}")
    
    # 2. 运行算法
    target_epsilon = 1e-2
    Q_approx = adaptive_randomized_range_finder(A, epsilon=target_epsilon)
    
    # 3. 验证结果
    found_rank = Q_approx.shape[1]
    print(f"算法计算出的秩 (Q的列数): {found_rank}")
    
    # 4. 验证近似误差 || (I - QQ*)A ||
    # I - QQ* 是投影到 Q 正交补空间的算子
    # 也就是 A 减去它在 Q 上的投影： A - Q(Q*A)
    diff = A - Q_approx @ (Q_approx.T @ A)
    error_norm = np.linalg.norm(diff, ord=2) # 谱范数
    
    print(f"近似误差 (Spectral Norm): {error_norm:.6f}")
    print(f"目标误差: {target_epsilon}")
    
    if error_norm < target_epsilon * 10: # 允许一定的随机浮动
        print(">> 测试通过：误差在可接受范围内。")
    else:
        print(">> 测试警告：误差偏大，请检查参数。")