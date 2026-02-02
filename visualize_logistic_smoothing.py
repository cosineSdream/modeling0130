"""
可视化logistic函数平滑过渡效果
对比阶跃函数 vs logistic函数
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

try:
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK CN', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass

# 时间范围（小时）
t_h = np.linspace(0, 24, 1000)

# 阶跃函数（原始）
stepping_func = np.where((t_h >= 9) & (t_h <= 18), 1.3, 0.8)

# Logistic平滑函数（新的）
morning_factor = 1.0 / (1.0 + np.exp(-3.0 * (t_h - 9.0)))      # 9点处上升
evening_factor = 1.0 / (1.0 + np.exp(3.0 * (t_h - 18.0)))      # 18点处下降
logistic_func = 0.8 + 0.5 * morning_factor * evening_factor

# 创建对比图
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 第一个图：两种函数对比
ax1 = axes[0]
ax1.plot(t_h, stepping_func, 'r-', linewidth=2.5, label='原始阶跃函数', marker='o', markersize=3, markevery=50)
ax1.plot(t_h, logistic_func, 'b-', linewidth=2.5, label='Logistic平滑函数')
ax1.fill_between(t_h, stepping_func, alpha=0.2, color='red', label='阶跃差异')
ax1.axvline(x=9, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.axvline(x=18, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.text(9, 1.35, '9:00\n上升', fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
ax1.text(18, 1.35, '18:00\n下降', fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
ax1.set_ylabel('强度调制因子', fontsize=12, fontweight='bold')
ax1.set_title('通话时间强度调制：阶跃 vs Logistic平滑', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 24)
ax1.set_ylim(0.7, 1.4)

# 第二个图：梯度（光滑性指标）
ax2 = axes[1]
# 计算数值梯度
stepping_grad = np.gradient(stepping_func, t_h)
logistic_grad = np.gradient(logistic_func, t_h)

ax2.plot(t_h, stepping_grad, 'r-', linewidth=2.5, label='阶跃函数梯度（不连续）', marker='o', markersize=3, markevery=50)
ax2.plot(t_h, logistic_grad, 'b-', linewidth=2.5, label='Logistic函数梯度（连续平滑）')
ax2.axvline(x=9, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axvline(x=18, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
ax2.set_xlabel('时刻 (小时)', fontsize=12, fontweight='bold')
ax2.set_ylabel('梯度 (dλ/dt)', fontsize=12, fontweight='bold')
ax2.set_title('梯度对比：阶跃函数在9:00和18:00处有无穷大梯度（尖峰）', fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 24)

plt.tight_layout()
plt.savefig('d:/machine_learning_vscode/0130mcm/logistic_smoothing_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Logistic平滑对比图已保存到: logistic_smoothing_comparison.png")

# 打印相关数据
print("\n" + "="*80)
print("通话强度调制函数平滑化效果分析")
print("="*80)
print("\n关键时刻的强度值对比:")
print(f"{'时刻':^10} | {'原始阶跃':^12} | {'Logistic平滑':^14} | {'差异':^10}")
print("-" * 60)

key_times = [8.0, 8.5, 9.0, 9.5, 12.0, 17.5, 18.0, 18.5, 19.0]
for t in key_times:
    idx = np.argmin(np.abs(t_h - t))
    step_val = stepping_func[idx]
    logistic_val = logistic_func[idx]
    diff = abs(step_val - logistic_val)
    print(f"{t:6.1f}:00 | {step_val:12.4f} | {logistic_val:14.4f} | {diff:10.4f}")

print("\n" + "="*80)
print("Logistic函数的优点:")
print("="*80)
print("1. 连续性：所有阶导数都存在且连续")
print("2. 光滑性：避免了阶跃函数的尖峰梯度")
print("3. 物理合理性：更符合实际的使用习惯渐进变化")
print("4. 数值稳定性：对微分方程求解更稳定")
print("\nLogistic函数形式:")
print("  morning_factor = 1 / (1 + exp(-3 × (t - 9)))")
print("  evening_factor = 1 / (1 + exp(3 × (t - 18)))")
print("  λ(t) = 0.8 + 0.5 × morning_factor × evening_factor")
print("="*80 + "\n")

plt.show()
