import time
from collections import OrderedDict

class PerformanceTimer:
    """
    一个用于分块计时和生成总耗时报告的上下文管理器。
    使用 time.perf_counter() 获得高精度计时。
    """
    # 使用 OrderedDict 保证输出顺序与执行顺序一致
    _timers = OrderedDict()

    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        # 进入 with 语句块时触发
        self.start_time = time.perf_counter()
        print(f"⏱️  开始: {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出 with 语句块时触发
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        
        # 记录到类变量中
        if self.name in PerformanceTimer._timers:
            # 如果同一个名字出现多次（比如在循环里），累加时间
            PerformanceTimer._timers[self.name] += elapsed_time
        else:
            PerformanceTimer._timers[self.name] = elapsed_time
            
        print(f"✅ 完成: {self.name} [耗时: {elapsed_time:.4f}s]")

    @classmethod
    def print_summary(cls):
        """打印所有计时块的汇总报告"""
        print("\n" + "="*40)
        print("📊 性能耗时统计报告")
        print("="*40)
        
        total_time = sum(cls._timers.values())
        
        for name, duration in cls._timers.items():
            # 计算百分比
            percent = (duration / total_time) * 100 if total_time > 0 else 0
            # 打印进度条效果
            bar_len = int(percent / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            
            print(f"{name:<20} | {duration:8.4f}s | {bar} {percent:5.1f}%")
            
        print("-" * 40)
        print(f"{'总计耗时':<20} | {total_time:8.4f}s")
        print("="*40 + "\n")
        
    @classmethod
    def reset(cls):
        """清空计时记录（如果要在同一个脚本跑多轮实验）"""
        cls._timers.clear()
        
        
        
# 假设你把上面的类保存为了 timer_utils.py
# from timer_utils import PerformanceTimer 

# if __name__ == "__main__":
#     import torch
    
#     # --- 全局计时开始 ---
#     with PerformanceTimer("整个程序流程"):
        
#         # 1. 数据准备阶段
#         with PerformanceTimer("1. 数据生成"):
#             torch.manual_seed(42)
#             z_t = torch.randn(3, 64, 64)
#             # 模拟一点耗时操作
#             time.sleep(0.1) 

#         # 2. 核心算法阶段
#         with PerformanceTimer("2. 随机 SVD 计算"):
#             # 这里调用你的函数
#             # 假设你已经定义了 randomized_svd
#             U, S, Vh = randomized_svd(z_t, epsilon=1e-2)
            
#         # 3. 验证阶段
#         with PerformanceTimer("3. 重建与验证"):
#             recon = U @ torch.diag_embed(S) @ Vh
#             err = torch.norm(z_t - recon)
#             print(f"   >> 误差: {err:.4f}")
#             # 模拟耗时
#             time.sleep(0.05)

#     # --- 最后打印报表 ---
#     PerformanceTimer.print_summary()