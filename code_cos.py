import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# ===============================
# 从表格中提取的数据
# ===============================
soc_table = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
R0_table = np.array([0.0629, 0.0614, 0.0641, 0.0630, 0.0613, 0.0615, 0.0610, 0.0612, 0.0602])
R1_table = np.array([0.0392, 0.0377, 0.0330, 0.0312, 0.0300, 0.0596, 0.0436, 0.0354, 0.0198])
C1_table = np.array([1075.17, 1287.61, 1290.36, 1268.09, 1249.34, 905.44, 880.62, 921.69, 1112.36])

# ===============================
# 创建三次样条插值函数
# ===============================
R0_spline = CubicSpline(soc_table, R0_table, bc_type='natural')
R1_spline = CubicSpline(soc_table, R1_table, bc_type='natural')
C1_spline = CubicSpline(soc_table, C1_table, bc_type='natural')

# ===============================
# OCV 参数
# ===============================
p1 = 0.56777875        # V
p2 = -1.328625e-11     # V
lambda1 = -0.21950625  # Ah^-1
lambda2 = 2.330        # Ah^-1

Qeff = 5.0             # Ah
Ptotal = 10.0          # W

# ===============================
# 初值
# ===============================
s0 = 1.0      # SOC
Vrc0 = 0.0    # 极化电压
x0 = [s0, Vrc0]

# ===============================
# ODE 定义（使用三次样条插值函数）
# ===============================
def odefun(t, x):
    s, Vrc = x
    
    # 限制SOC范围，避免插值超出边界
    s_clipped = np.clip(s, 0.1, 0.9)
    
    # 从三次样条插值函数获取当前SOC对应的参数
    R0 = float(R0_spline(s_clipped))
    R1 = float(R1_spline(s_clipped))
    C1 = float(C1_spline(s_clipped))
    
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
        + 4.2
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
# 事件函数：SOC 到 0.1 停止（因为插值范围是0.1-0.9）
# ===============================
def stop_event(t, x):
    return x[0] - 0.1

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

for i in range(len(t)):
    s_val = s[i]
    Vrc_val = Vrc[i]
    
    s_clipped = np.clip(s_val, 0.1, 0.9)
    R0_val = float(R0_spline(s_clipped))
    R1_val = float(R1_spline(s_clipped))
    C1_val = float(C1_spline(s_clipped))
    
    R0_val = max(R0_val, 1e-6)
    R1_val = max(R1_val, 1e-6)
    C1_val = max(C1_val, 1e-6)
    
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

IL_array = np.array(IL_array)
Voc_array = np.array(Voc_array)
Vt_array = np.array(Vt_array)
R0_array = np.array(R0_array)
R1_array = np.array(R1_array)
C1_array = np.array(C1_array)

# ===============================
# 绘图
# ===============================
fig = plt.figure(figsize=(16, 12))

# 创建3x3的子图布局
# 第一行：SOC, 电流, 电压
ax1 = plt.subplot(3, 3, 1)
ax1.plot(t / 60, s, linewidth=2, color='b')
ax1.grid(True)
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("SOC")
ax1.set_title("SOC vs Time")

ax2 = plt.subplot(3, 3, 2)
ax2.plot(t / 60, IL_array, linewidth=2, color='r')
ax2.grid(True)
ax2.set_xlabel("Time (min)")
ax2.set_ylabel("Current (A)")
ax2.set_title("Discharge Current vs Time")

ax3 = plt.subplot(3, 3, 3)
ax3.plot(t / 60, Voc_array, linewidth=2, label='Voc', color='g')
ax3.plot(t / 60, Vt_array, linewidth=2, label='Vt (Terminal)', color='orange')
ax3.plot(t / 60, Vrc, linewidth=2, label='Vrc (Polarization)', color='purple')
ax3.grid(True)
ax3.set_xlabel("Time (min)")
ax3.set_ylabel("Voltage (V)")
ax3.set_title("Voltages vs Time")
ax3.legend()

# 第二行：R0, R1, C1 vs Time
ax4 = plt.subplot(3, 3, 4)
ax4.plot(t / 60, R0_array * 1000, linewidth=2, color='darkblue')
ax4.grid(True)
ax4.set_xlabel("Time (min)")
ax4.set_ylabel("R0 (mΩ)")
ax4.set_title("R0 vs Time")

ax5 = plt.subplot(3, 3, 5)
ax5.plot(t / 60, R1_array * 1000, linewidth=2, color='darkred')
ax5.grid(True)
ax5.set_xlabel("Time (min)")
ax5.set_ylabel("R1 (mΩ)")
ax5.set_title("R1 vs Time")

ax6 = plt.subplot(3, 3, 6)
ax6.plot(t / 60, C1_array, linewidth=2, color='darkgreen')
ax6.grid(True)
ax6.set_xlabel("Time (min)")
ax6.set_ylabel("C1 (F)")
ax6.set_title("C1 vs Time")

# 第三行：参数 vs SOC（显示插值曲线和原始数据点）
soc_fine = np.linspace(0.1, 0.9, 200)

ax7 = plt.subplot(3, 3, 7)
ax7.plot(soc_fine, R0_spline(soc_fine) * 1000, linewidth=2, color='blue', label='Cubic Spline')
ax7.scatter(soc_table, R0_table * 1000, color='red', s=50, zorder=5, label='Data Points')
ax7.grid(True)
ax7.set_xlabel("SOC")
ax7.set_ylabel("R0 (mΩ)")
ax7.set_title("R0 vs SOC (Cubic Spline Interpolation)")
ax7.legend()

ax8 = plt.subplot(3, 3, 8)
ax8.plot(soc_fine, R1_spline(soc_fine) * 1000, linewidth=2, color='red', label='Cubic Spline')
ax8.scatter(soc_table, R1_table * 1000, color='blue', s=50, zorder=5, label='Data Points')
ax8.grid(True)
ax8.set_xlabel("SOC")
ax8.set_ylabel("R1 (mΩ)")
ax8.set_title("R1 vs SOC (Cubic Spline Interpolation)")
ax8.legend()

ax9 = plt.subplot(3, 3, 9)
ax9.plot(soc_fine, C1_spline(soc_fine), linewidth=2, color='green', label='Cubic Spline')
ax9.scatter(soc_table, C1_table, color='orange', s=50, zorder=5, label='Data Points')
ax9.grid(True)
ax9.set_xlabel("SOC")
ax9.set_ylabel("C1 (F)")
ax9.set_title("C1 vs SOC (Cubic Spline Interpolation)")
ax9.legend()

plt.tight_layout()
plt.show()

# ===============================
# 输出统计信息
# ===============================
print("=" * 60)
print("仿真结果统计")
print("=" * 60)
print(f"放电时间: {t[-1]/60:.2f} 分钟 ({t[-1]/3600:.2f} 小时)")
print(f"初始SOC: {s[0]:.4f}")
print(f"最终SOC: {s[-1]:.4f}")
print(f"平均电流: {np.mean(IL_array):.4f} A")
print(f"最大电流: {np.max(IL_array):.4f} A")
print(f"最小电流: {np.min(IL_array):.4f} A")
print(f"初始端电压: {Vt_array[0]:.4f} V")
print(f"最终端电压: {Vt_array[-1]:.4f} V")
print(f"平均端电压: {np.mean(Vt_array):.4f} V")
print("=" * 60)