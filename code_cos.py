import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ===============================
# 从表格中提取的数据
# ===============================
soc_table = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
R0_table = np.array([0.0602, 0.0612, 0.0610, 0.0615, 0.0613, 0.0630, 0.0641, 0.0614, 0.0629])
R1_table = np.array([0.0198, 0.0354, 0.0436, 0.0596, 0.0300, 0.0312, 0.0330, 0.0377, 0.0392])
C1_table = np.array([1112.36, 921.69, 880.62, 905.44, 1249.34, 1268.09, 1290.36, 1287.61, 1075.17])

# ===============================
# 创建插值函数（线性插值）
# ===============================
R0_interp = interp1d(soc_table, R0_table, kind='linear', bounds_error=False, fill_value='extrapolate')
R1_interp = interp1d(soc_table, R1_table, kind='linear', bounds_error=False, fill_value='extrapolate')
C1_interp = interp1d(soc_table, C1_table, kind='linear', bounds_error=False, fill_value='extrapolate')

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
# ODE 定义（使用插值函数）
# ===============================
def odefun(t, x):
    s, Vrc = x
    
    # 限制SOC范围，避免插值超出边界
    s_clipped = np.clip(s, 0.05, 1.0)
    
    # 从插值函数获取当前SOC对应的参数
    R0 = float(R0_interp(s_clipped))
    R1 = float(R1_interp(s_clipped))
    C1 = float(C1_interp(s_clipped))
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

for i in range(len(t)):
    s_val = s[i]
    Vrc_val = Vrc[i]
    
    s_clipped = np.clip(s_val, 0.05, 1.0)
    R0_val = float(R0_interp(s_clipped))
    R1_val = float(R1_interp(s_clipped))
    
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

IL_array = np.array(IL_array)
Voc_array = np.array(Voc_array)
Vt_array = np.array(Vt_array)
R0_array = np.array(R0_array)
R1_array = np.array(R1_array)

# ===============================
# 绘图
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SOC vs time
axes[0, 0].plot(t / 60, s, linewidth=2, color='b')
axes[0, 0].grid(True)
axes[0, 0].set_xlabel("Time (min)")
axes[0, 0].set_ylabel("SOC")
axes[0, 0].set_title("SOC vs Time under Constant Power Load (P = 10 W)")

# 电流 vs time
axes[0, 1].plot(t / 60, IL_array, linewidth=2, color='r')
axes[0, 1].grid(True)
axes[0, 1].set_xlabel("Time (min)")
axes[0, 1].set_ylabel("Current (A)")
axes[0, 1].set_title("Discharge Current vs Time")

# 电压 vs time
axes[1, 0].plot(t / 60, Voc_array, linewidth=2, label='Voc', color='g')
axes[1, 0].plot(t / 60, Vt_array, linewidth=2, label='Vt (Terminal)', color='orange')
axes[1, 0].grid(True)
axes[1, 0].set_xlabel("Time (min)")
axes[1, 0].set_ylabel("Voltage (V)")
axes[1, 0].set_title("Voltages vs Time")
axes[1, 0].legend()

# R0, R1 vs SOC
axes[1, 1].plot(s, R0_array, linewidth=2, label='R0', color='purple')
axes[1, 1].plot(s, R1_array, linewidth=2, label='R1', color='brown')
axes[1, 1].grid(True)
axes[1, 1].set_xlabel("SOC")
axes[1, 1].set_ylabel("Resistance (Ω)")
axes[1, 1].set_title("R0 and R1 vs SOC (from interpolation)")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"放电时间: {t[-1]/60:.2f} 分钟")
print(f"最终SOC: {s[-1]:.4f}")