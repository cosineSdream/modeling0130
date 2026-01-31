import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ===============================
# 从表格中提取的数据
# ===============================
import numpy as np

soc_table = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
R0_table = np.array([0.0344, 0.0324, 0.0306, 0.0296, 0.0284, 0.0279, 0.0271, 0.0263, 0.0251])
R1_table = np.array([0.0750, 0.0487, 0.0392, 0.0367, 0.0317, 0.0312, 0.0287, 0.0351, 0.0280])
C1_table = np.array([427.09, 520.73, 567.06, 618.50, 649.01, 687.74, 720.09, 747.04, 769.39])

# ===============================
# 三次多项式拟合
# ===============================
# 拟合函数: y = a*x^3 + b*x^2 + c*x + d
R0_coeffs = np.polyfit(soc_table, R0_table, 3)
R1_coeffs = np.polyfit(soc_table, R1_table, 3)
C1_coeffs = np.polyfit(soc_table, C1_table, 3)

# 创建多项式函数
R0_poly = np.poly1d(R0_coeffs)
R1_poly = np.poly1d(R1_coeffs)
C1_poly = np.poly1d(C1_coeffs)

# 打印拟合结果
print("=" * 70)
print("三次多项式拟合结果")
print("=" * 70)
print(f"R0(s) = {R0_coeffs[0]:.8f}*s^3 + {R0_coeffs[1]:.8f}*s^2 + {R0_coeffs[2]:.8f}*s + {R0_coeffs[3]:.8f}")
print(f"R1(s) = {R1_coeffs[0]:.8f}*s^3 + {R1_coeffs[1]:.8f}*s^2 + {R1_coeffs[2]:.8f}*s + {R1_coeffs[3]:.8f}")
print(f"C1(s) = {C1_coeffs[0]:.6f}*s^3 + {C1_coeffs[1]:.6f}*s^2 + {C1_coeffs[2]:.6f}*s + {C1_coeffs[3]:.6f}")
print("=" * 70)

# 计算拟合优度 R²
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# 计算均方根误差 RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

R0_fit = R0_poly(soc_table)
R1_fit = R1_poly(soc_table)
C1_fit = C1_poly(soc_table)

print(f"\nR0 拟合优度 R² = {r_squared(R0_table, R0_fit):.6f}, RMSE = {rmse(R0_table, R0_fit)*1000:.4f} mΩ")
print(f"R1 拟合优度 R² = {r_squared(R1_table, R1_fit):.6f}, RMSE = {rmse(R1_table, R1_fit)*1000:.4f} mΩ")
print(f"C1 拟合优度 R² = {r_squared(C1_table, C1_fit):.6f}, RMSE = {rmse(C1_table, C1_fit):.4f} F")
print("=" * 70)

# ===============================
# OCV 参数
# ===============================
p1 = 0.56777875        # V
p2 = -1.328625e-11     # V
lambda1 = -0.21950625*2.25 # Ah^-1
lambda2 = 2.330*2.25        # Ah^-1

Qeff = 5.0             # Ah
Ptotal = 10.0          # W

# ===============================
# 初值
# ===============================
s0 = 1.0      # SOC
Vrc0 = 0.0    # 极化电压
x0 = [s0, Vrc0]

# ===============================
# ODE 定义（使用三次多项式拟合函数）
# ===============================
def odefun(t, x):
    s, Vrc = x
    
    # 限制SOC范围
    s_clipped = np.clip(s, 0.05, 1.0)
    
    # 从三次多项式函数获取当前SOC对应的参数
    R0 = float(R0_poly(s_clipped))
    R1 = float(R1_poly(s_clipped))
    C1 = float(C1_poly(s_clipped))
    
    # 确保参数为正值（物理意义）
    R0 = max(R0, 1e-6)
    R1 = max(R1, 1e-6)
    C1 = max(C1, 1e-6)
    
    gamma = R1 * C1  # 时间常数 (s)

    # Voc(s)
    q = (1 - s) * Qeff  # Ah
    Voc = (
        p1 * (np.exp(lambda1 * q) - 1)
        + p2 * (np.exp(lambda2 * q) - 1)
        + 3.65
    )

    # 恒功率条件下解电流
    # P = IL * (Voc - Vrc - R0 * IL)
    Vt = Voc - Vrc
    disc = Vt**2 - 4 * R0 * Ptotal
    if disc < 0:
        disc = 0.0  # 数值保护

    # 选较小的正根
    IL = (Vt - np.sqrt(disc)) / (2 * R0)

    # 状态方程
    dsdt = -IL / Qeff / 3600.0   # 1/s
    dVrcdt = (IL * R1 - Vrc) / gamma

    return [dsdt, dVrcdt]

# ===============================
# 事件函数：SOC 到 0.05 停止
# ===============================
def stop_event(t, x):
    return x[0] - 0.05

stop_event.terminal = True
stop_event.direction = -1

# ===============================
# 求解
# ===============================
t_span = (0, 5 * 3600)  # 秒
sol = solve_ivp(
    odefun,
    t_span,
    x0,
    method='RK45',
    rtol=1e-8,
    atol=1e-10,
    events=stop_event,
    dense_output=True
)

t = sol.t
s = sol.y[0]
Vrc = sol.y[1]

# ===============================
# 计算其他物理量用于绘图
# ===============================
IL_array = []
Voc_array = []
Vt_array = []
R0_array = []
R1_array = []
C1_array = []
gamma_array = []

for i in range(len(t)):
    s_val = s[i]
    Vrc_val = Vrc[i]
    
    s_clipped = np.clip(s_val, 0.05, 1.0)
    R0_val = float(R0_poly(s_clipped))
    R1_val = float(R1_poly(s_clipped))
    C1_val = float(C1_poly(s_clipped))
    
    R0_val = max(R0_val, 1e-6)
    R1_val = max(R1_val, 1e-6)
    C1_val = max(C1_val, 1e-6)
    
    gamma_val = R1_val * C1_val
    
    q = (1 - s_val) * Qeff
    Voc_val = (
        p1 * (np.exp(lambda1 * q) - 1)
        + p2 * (np.exp(lambda2 * q) - 1)
        + 4.2
    )
    
    Vt_val = Voc_val - Vrc_val
    disc = Vt_val**2 - 4 * R0_val * Ptotal
    if disc < 0:
        disc = 0.0
    IL_val = (Vt_val - np.sqrt(disc)) / (2 * R0_val)
    
    IL_array.append(IL_val)
    Voc_array.append(Voc_val)
    Vt_array.append(Vt_val)
    R0_array.append(R0_val)
    R1_array.append(R1_val)
    C1_array.append(C1_val)
    gamma_array.append(gamma_val)

IL_array = np.array(IL_array)
Voc_array = np.array(Voc_array)
Vt_array = np.array(Vt_array)
R0_array = np.array(R0_array)
R1_array = np.array(R1_array)
C1_array = np.array(C1_array)
gamma_array = np.array(gamma_array)

# ===============================
# 绘图
# ===============================
fig = plt.figure(figsize=(18, 12))

# 创建4x3的子图布局
# 第一行：SOC, 电流, 电压
ax1 = plt.subplot(4, 3, 1)
ax1.plot(t / 60, s, linewidth=2, color='b')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel("Time (min)", fontsize=10)
ax1.set_ylabel("SOC", fontsize=10)
ax1.set_title("SOC vs Time under Constant Power Load (P = 10 W)", fontsize=11)

ax2 = plt.subplot(4, 3, 2)
ax2.plot(t / 60, IL_array, linewidth=2, color='r')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel("Time (min)", fontsize=10)
ax2.set_ylabel("Current (A)", fontsize=10)
ax2.set_title("Discharge Current vs Time", fontsize=11)

ax3 = plt.subplot(4, 3, 3)
ax3.plot(t / 60, Voc_array, linewidth=2, label='Voc', color='g')
ax3.plot(t / 60, Vt_array, linewidth=2, label='Vt (Terminal)', color='orange')
ax3.plot(t / 60, Vrc, linewidth=2, label='Vrc (Polarization)', color='purple')
ax3.grid(True, alpha=0.3)
ax3.set_xlabel("Time (min)", fontsize=10)
ax3.set_ylabel("Voltage (V)", fontsize=10)
ax3.set_title("Voltages vs Time", fontsize=11)
ax3.legend(fontsize=8)

# 第二行：R0, R1, C1 vs Time
ax4 = plt.subplot(4, 3, 4)
ax4.plot(t / 60, R0_array * 1000, linewidth=2, color='darkblue')
ax4.grid(True, alpha=0.3)
ax4.set_xlabel("Time (min)", fontsize=10)
ax4.set_ylabel("R0 (mΩ)", fontsize=10)
ax4.set_title("R0 vs Time", fontsize=11)

ax5 = plt.subplot(4, 3, 5)
ax5.plot(t / 60, R1_array * 1000, linewidth=2, color='darkred')
ax5.grid(True, alpha=0.3)
ax5.set_xlabel("Time (min)", fontsize=10)
ax5.set_ylabel("R1 (mΩ)", fontsize=10)
ax5.set_title("R1 vs Time", fontsize=11)

ax6 = plt.subplot(4, 3, 6)
ax6.plot(t / 60, C1_array, linewidth=2, color='darkgreen')
ax6.grid(True, alpha=0.3)
ax6.set_xlabel("Time (min)", fontsize=10)
ax6.set_ylabel("C1 (F)", fontsize=10)
ax6.set_title("C1 vs Time", fontsize=11)

# 第三行：参数 vs SOC（显示拟合曲线和原始数据点）
soc_fine = np.linspace(0.1, 0.9, 200)

ax7 = plt.subplot(4, 3, 7)
ax7.plot(soc_fine, R0_poly(soc_fine) * 1000, linewidth=2, color='blue', label='Cubic Fit')
ax7.scatter(soc_table, R0_table * 1000, color='red', s=60, zorder=5, label='Data Points', edgecolor='black', linewidth=0.5)
ax7.grid(True, alpha=0.3)
ax7.set_xlabel("SOC", fontsize=10)
ax7.set_ylabel("R0 (mΩ)", fontsize=10)
ax7.set_title(f"R0 vs SOC (R² = {r_squared(R0_table, R0_fit):.4f})", fontsize=11)
ax7.legend(fontsize=8)

ax8 = plt.subplot(4, 3, 8)
ax8.plot(soc_fine, R1_poly(soc_fine) * 1000, linewidth=2, color='red', label='Cubic Fit')
ax8.scatter(soc_table, R1_table * 1000, color='blue', s=60, zorder=5, label='Data Points', edgecolor='black', linewidth=0.5)
ax8.grid(True, alpha=0.3)
ax8.set_xlabel("SOC", fontsize=10)
ax8.set_ylabel("R1 (mΩ)", fontsize=10)
ax8.set_title(f"R1 vs SOC (R² = {r_squared(R1_table, R1_fit):.4f})", fontsize=11)
ax8.legend(fontsize=8)

ax9 = plt.subplot(4, 3, 9)
ax9.plot(soc_fine, C1_poly(soc_fine), linewidth=2, color='green', label='Cubic Fit')
ax9.scatter(soc_table, C1_table, color='orange', s=60, zorder=5, label='Data Points', edgecolor='black', linewidth=0.5)
ax9.grid(True, alpha=0.3)
ax9.set_xlabel("SOC", fontsize=10)
ax9.set_ylabel("C1 (F)", fontsize=10)
ax9.set_title(f"C1 vs SOC (R² = {r_squared(C1_table, C1_fit):.4f})", fontsize=11)
ax9.legend(fontsize=8)

# 第四行：时间常数和功率
ax10 = plt.subplot(4, 3, 10)
ax10.plot(t / 60, gamma_array, linewidth=2, color='darkviolet')
ax10.grid(True, alpha=0.3)
ax10.set_xlabel("Time (min)", fontsize=10)
ax10.set_ylabel("γ = R1·C1 (s)", fontsize=10)
ax10.set_title("Time Constant vs Time", fontsize=11)

ax11 = plt.subplot(4, 3, 11)
power_array = IL_array * Vt_array
ax11.plot(t / 60, power_array, linewidth=2, color='darkorange')
ax11.axhline(y=Ptotal, color='k', linestyle='--', linewidth=1, label=f'Target: {Ptotal} W')
ax11.grid(True, alpha=0.3)
ax11.set_xlabel("Time (min)", fontsize=10)
ax11.set_ylabel("Power (W)", fontsize=10)
ax11.set_title("Output Power vs Time", fontsize=11)
ax11.legend(fontsize=8)

ax12 = plt.subplot(4, 3, 12)
energy_cumulative = np.cumsum(power_array * np.diff(np.concatenate(([0], t)))) / 3600  # Wh
ax12.plot(t / 60, energy_cumulative, linewidth=2, color='brown')
ax12.grid(True, alpha=0.3)
ax12.set_xlabel("Time (min)", fontsize=10)
ax12.set_ylabel("Cumulative Energy (Wh)", fontsize=10)
ax12.set_title("Cumulative Energy Output vs Time", fontsize=11)

plt.tight_layout()
plt.show()

# ===============================
# 输出统计信息
# ===============================
print("\n" + "=" * 70)
print("仿真结果统计")
print("=" * 70)
print(f"放电时间: {t[-1]/60:.2f} 分钟 ({t[-1]/3600:.2f} 小时)")
print(f"初始SOC: {s[0]:.4f}")
print(f"最终SOC: {s[-1]:.4f}")
print(f"释放容量: {(s[0] - s[-1]) * Qeff:.4f} Ah")
print(f"\n电流统计:")
print(f"  平均电流: {np.mean(IL_array):.4f} A")
print(f"  最大电流: {np.max(IL_array):.4f} A")
print(f"  最小电流: {np.min(IL_array):.4f} A")
print(f"\n电压统计:")
print(f"  初始开路电压: {Voc_array[0]:.4f} V")
print(f"  最终开路电压: {Voc_array[-1]:.4f} V")
print(f"  初始端电压: {Vt_array[0]:.4f} V")
print(f"  最终端电压: {Vt_array[-1]:.4f} V")
print(f"  平均端电压: {np.mean(Vt_array):.4f} V")
print(f"\n能量统计:")
print(f"  总能量输出: {energy_cumulative[-1]:.4f} Wh")
print(f"  目标功率: {Ptotal:.2f} W")
print(f"  平均功率误差: {np.mean(np.abs(power_array - Ptotal)):.4f} W")
print("=" * 70)

# ===============================
# 新增图：Voc vs Discharged Capacity & Voc vs SOC
# ===============================

# 放电电量 q (Ah)
q_array = (1 - s) * Qeff

fig2 = plt.figure(figsize=(12, 5))

# 子图 1：Voc vs 放电电量
ax_q = plt.subplot(1, 2, 1)
ax_q.plot(q_array, Voc_array, linewidth=2, color='teal')
ax_q.grid(True, alpha=0.3)
ax_q.set_xlabel("Discharged Capacity q (Ah)", fontsize=11)
ax_q.set_ylabel("Open Circuit Voltage Voc (V)", fontsize=11)
ax_q.set_title("Voc vs Discharged Capacity", fontsize=12)

# 子图 2：Voc vs SOC
ax_soc = plt.subplot(1, 2, 2)
ax_soc.plot(s, Voc_array, linewidth=2, color='darkorange')
ax_soc.grid(True, alpha=0.3)
ax_soc.set_xlabel("State of Charge (SOC)", fontsize=11)
ax_soc.set_ylabel("Open Circuit Voltage Voc (V)", fontsize=11)
ax_soc.set_title("Voc vs SOC", fontsize=12)
ax_soc.invert_xaxis()  # 放电方向：SOC 从 1 → 0（论文中更直观）

plt.tight_layout()
plt.show()
