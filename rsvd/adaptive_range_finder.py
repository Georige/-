import numpy as np

def adaptive_randomized_range_finder(A: np.ndarray, epsilon: float, r: int = 10) -> np.ndarray:
    """
    实现算法 4.2: 自适应随机化 Range Finder。
    
    该函数计算矩阵 A 的正交基 Q，使得近似误差在概率上小于 epsilon。
    
    参数:
        A (np.ndarray): 输入矩阵，形状为 (m, n)。
        epsilon (float): 容差阈值。
        r (int): 块大小/过采样参数，默认为 10。
        
    返回:
        Q (np.ndarray): 正交矩阵，形状为 (m, k)，k 是自适应确定的秩。
    """
    
    # 获取矩阵维度 m (行数) x n (列数)
    m, n = A.shape
    
    # --- 步骤 1: 初始化 ---
    # Draw standard Gaussian vectors (生成 r 个标准高斯随机向量)
    # 形状为 (n, r)
    Omega = np.random.normal(size=(n, r))
    
    # --- 步骤 2: 初始采样 ---
    # Compute Y = A * Omega
    # 此时 Y 是一个包含 r 个向量的列表，每个向量形状为 (m,)
    # 注意：为了方便后续动态添加，我们使用 Python List 存储向量，而不是固定矩阵
    Y = [A @ Omega[:, i] for i in range(r)]
    
    # --- 步骤 3 & 4: 初始化循环变量 ---
    j = 0
    Q = []  # Q 初始为空列表，后续会将基向量 append 进去
    
    # 计算阈值 limit
    # 公式: epsilon / (10 * sqrt(2/pi))
    # np.sqrt(2 / np.pi) 约等于 0.798
    limit = epsilon / (10 * np.sqrt(2 / np.pi))
    
    # --- 步骤 5: While 循环条件 ---
    # 检查当前窗口内 r 个向量的最大范数是否超过阈值
    # Y[j : j+r] 对应算法中的 y^(j+1)...y^(j+r)
    while True:
        # 获取当前窗口内的向量
        current_window = Y[j : j+r]
        
        # 计算窗口内每个向量的 L2 范数
        norms = [np.linalg.norm(y) for y in current_window]
        max_norm = np.max(norms)
        
        # 如果最大范数小于阈值，说明剩余的能量已经很小了，停止循环
        if max_norm <= limit:
            break
            
        # --- 步骤 6: 索引递增 ---
        # 算法中 j = j + 1。在 Python 0-indexed 语境下，
        # 我们当前处理的向量索引就是 j。
        
        # --- 步骤 7: 投影 (覆盖 y^(j)) ---
        # 这一步在理论上是 (I - Q Q*)y。
        # 但在算法步骤 13 中我们已经对后续向量做过正交化了。
        # 为了数值稳定性，我们在这里再次确保 y_j 与之前的 Q 正交（可选，但推荐）
        y_current = Y[j]
        for q_prev in Q:
            y_current = y_current - q_prev * np.dot(q_prev, y_current)
        
        # --- 步骤 8: 归一化 ---
        # q^(j) = y^(j) / ||y^(j)||
        norm_y = np.linalg.norm(y_current)
        
        # 防止除以零（极罕见情况，作为系统防御性编程）
        if norm_y < 1e-15:
            # 如果向量范数极小，说明已经没有新信息，可以跳过或中断
            break
            
        q_new = y_current / norm_y
        
        # --- 步骤 9: 更新 Q ---
        Q.append(q_new)
        
        # --- 步骤 10: 生成新的高斯向量 ---
        # Draw standard Gaussian vector of length n
        omega_new = np.random.normal(size=n)
        
        # --- 步骤 11: 计算新样本并投影 ---
        # y^(j+r) = (I - Q Q*) A omega_new
        # 先计算 A * omega
        y_new = A @ omega_new
        
        # 立即对现有的 Q 进行正交化投影
        # y_new = y_new - sum(q * <q, y_new>)
        for q in Q:
            y_new = y_new - q * np.dot(q, y_new)
            
        # 将新样本加入列表 Y
        Y.append(y_new)
        
        # --- 步骤 12 & 13: 更新前瞻窗口内的向量 (Re-orthogonalization) ---
        # 对当前窗口内剩余的 r-1 个向量，减去它们在 q_new 上的投影
        # for i = (j+1) to (j+r-1)
        for i in range(j + 1, j + r):
            # Overwrite y^(i) by y^(i) - q^(j) <q^(j), y^(i)>
            # 这是一个典型的 Gram-Schmidt 步骤
            Y[i] = Y[i] - q_new * np.dot(q_new, Y[i])
            
        # 准备下一次迭代
        j += 1

    # --- 步骤 16: 构建最终矩阵 ---
    # 将向量列表转换为矩阵 (m, k)
    if not Q:
        return np.zeros((m, 0))
    
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