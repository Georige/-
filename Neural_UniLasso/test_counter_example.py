import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. 动态导入你的库 ---
# 尝试从当前目录导入你的函数式 API
try:
    from uni_lasso import cv_unilasso, extract_cv, predict
    print("成功导入 uni_lasso 模块")
except ImportError:
    try:
        from unilasso import cv_unilasso, extract_cv, predict
        print("成功导入 unilasso 模块")
    except ImportError:
        raise ImportError("无法导入 uni_lasso 或 unilasso，请检查文件名是否为 uni_lasso.py")

def generate_trap_data(n=100, p=20, sigma=0.5):
    """
    生成论文 Section 8 (Counter-example) 的数据 。
    强制要求 x2 的单变量系数为正，以触发 UniLasso 的陷阱。
    """
    max_retries = 100
    for i in range(max_retries):
        # 1. 生成特征
        # x1 ~ N(0,1)
        # x2 = x1 + N(0,1) 
        # 这里的噪声导致 x2 与 x1 高度相关
        X = np.random.normal(0, 1, (n, p))
        X[:, 1] = X[:, 0] + np.random.normal(0, 1, n)
        
        # 2. 真实系数
        # beta = (1, -0.5, 0, ...) 
        true_beta = np.zeros(p)
        true_beta[0] = 1.0
        true_beta[1] = -0.5
        
        # 3. 生成 Y
        # error SD = 0.5 
        y = np.dot(X, true_beta) + np.random.normal(0, sigma, n)
        
        # 4. 验证陷阱条件
        # 我们必须确保 x2 的单变量相关性是正的，而它的真实系数是负的
        # 这样 UniLasso 才会因为符号限制而犯错
        corr_x2_y = np.corrcoef(X[:, 1], y)[0, 1]
        
        if corr_x2_y > 0.05: # 设定一个小的正阈值确保陷阱形成
            print(f"--> 在第 {i+1} 次尝试中生成了有效陷阱数据 (Corr(x2, y)={corr_x2_y:.3f})")
            return X, y, true_beta
            
    raise RuntimeError("无法生成满足条件的陷阱数据，请检查数据生成逻辑。")

def run_experiment():
    print("\n>>> 开始运行 UniLasso 反例实验 (函数式 API 版本) <<<")
    
    # 1. 准备数据
    X, y, true_beta = generate_trap_data(n=100, p=20)
    
    # 划分训练集和测试集 (论文中使用了 70/30 或直接模拟)
    # 这里我们划分以便计算 Test MSE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"数据准备完毕: Train n={len(y_train)}, Test n={len(y_test)}")

    # 2. 检查单变量视角 (陷阱验证)
    reg_x2 = LinearRegression().fit(X_train[:, 1].reshape(-1, 1), y_train)
    uni_beta_2 = reg_x2.coef_[0]
    print(f"\n[陷阱状态]")
    print(f"x2 真实多变量系数: {true_beta[1]}")
    print(f"x2 单变量回归系数: {uni_beta_2:.4f} (应该是正数)")
    
    # 3. 运行 Baseline: Sklearn Lasso
    print("\n--- Model 1: Standard Lasso (Sklearn) ---")
    lasso = LassoCV(cv=5).fit(X_train, y_train)
    mse_lasso = mean_squared_error(y_test, lasso.predict(X_test))
    print(f"Lasso Test MSE: {mse_lasso:.4f}")
    print(f"Lasso x2 Coef : {lasso.coef_[1]:.4f} (应该接近 -0.5)")

    # 4. 运行 UniLasso (你的库)
    print("\n--- Model 2: UniLasso (Your Implementation) ---")
    # 使用你提供的 CV 函数接口
    # 假设 cv_unilasso 返回一个包含路径和 CV 结果的对象
    try:
        cv_fit = cv_unilasso(X_train, y_train, family='gaussian')
        
        # 提取最佳模型
        best_model = extract_cv(cv_fit)
        
        # 预测
        y_pred_uni = predict(best_model, X_test)
        mse_uni = mean_squared_error(y_test, y_pred_uni)
        
        # 获取系数
        # 注意：需要确认 best_model.coefs 的格式 (是列表、数组还是字典)
        uni_coefs = best_model.coefs 
        # 假设 coefs 是一个数组，顺序对应特征 0, 1, 2...
        # 如果包含截距，通常截距是第一个，或者分开存储。
        # 这里假设 coefs 长度等于 p (不含截距) 或 p+1 (含截距)
        
        print(f"UniLasso Test MSE: {mse_uni:.4f}")
        
        # 尝试寻找 x2 的系数
        # 如果系数包含截距，x2 可能是索引 2；如果不含，是索引 1
        x2_uni_coef = 0.0
        if len(uni_coefs) == 20:
            x2_uni_coef = uni_coefs[1]
        elif len(uni_coefs) == 21: # 包含截距
            x2_uni_coef = uni_coefs[2]
        else:
            # 简单的 fallback，打印前几个
            print(f"UniLasso Coefs (raw): {uni_coefs[:5]}")
            x2_uni_coef = uni_coefs[1] if len(uni_coefs) > 1 else 0

        print(f"UniLasso x2 Coef: {x2_uni_coef:.4f}")

        # 5. 结论
        print("\n--- 实验结论 ---")
        if mse_uni > mse_lasso and (x2_uni_coef >= 0):
            print("【验证成功】")
            print("UniLasso 掉入了陷阱！")
            print("原因：它看到 x2 的单变量系数为正，因此强制将其多变量系数设为非负（通常被压缩为0）。")
            print("结果：丢失了真实的负信号，导致 MSE 高于普通 Lasso。")
            print("这完美复现了论文 [cite: 157, 158] 描述的 'Achilles heel'。")
        else:
            print("【结果不明确】")
            print("UniLasso 似乎表现得还不错，或者系数并未被压缩为0。请检查非负约束逻辑。")

    except Exception as e:
        print(f"UniLasso 运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_experiment()