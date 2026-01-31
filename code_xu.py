import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
# 配置字体
matplotlib.rc("font", family='Microsoft YaHei', weight="bold")
# 参数
p1 = 0.56777875
p2 = -1.328625e-11
lambda1 = -0.21950625
lambda2 = 2.330
Vocmax = 4.2
Qeff = 7.5  # 有效电荷容量，单位 Ah
R0 = 0.1  # ohm
R1 = 0.04  # ohm
C1 = 1000  # F
P = 5  # W

def Voc_func(q):
    return p1 * (np.exp(lambda1 * q) - 1) + p2 * (np.exp(lambda2 * q) - 1) + Vocmax

def battery_ode(t, y):
    SOC, Vrc = y
    q = (1 - SOC) * Qeff
    Voc = Voc_func(q)
    
    # 解 IL 从 P = VL * IL = 10, VL = Voc - IL*R0 - Vrc
    # -R0 * IL^2 + (Voc - Vrc) * IL - 10 = 0
    a = -R0
    b = Voc - Vrc
    c = -P
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        IL = 0  # 如果无解，设IL=0
    else:
        IL = (-b + np.sqrt(discriminant)) / (2 * a)
    
    dSOC_dt = -IL / Qeff
    dVrc_dt = (IL * R1 - Vrc) / (R1 * C1)
    return [dSOC_dt, dVrc_dt]

def soc_zero(t, y):
    return y[0] - 0.8  # 停止当SOC=0.8

soc_zero.terminal = True
soc_zero.direction = -1

# 初始条件
y0 = [1.0, 0.0]  # SOC=1, Vrc=0
t_span = (0, 100)  # 足够长的时间

# 求解
sol = solve_ivp(battery_ode, t_span, y0, events=soc_zero, max_step=10)

# 计算 q 和 Voc
q = (1 - sol.y[0]) * Qeff
Voc = Voc_func(q)

# 绘制图表
plt.figure(figsize=(15, 5))

# 子图1: Voc vs 放电电量 (q)
plt.subplot(1, 3, 1)
plt.plot(q, Voc)
plt.xlabel('放电电量 (Ah)')
plt.ylabel('Voc (V)')
plt.title('Voc vs 放电电量')
plt.grid(True)

# 子图2: SOC vs Voc (静态关系)
SOC_range = np.linspace(0.8, 1, 10000)
q_range = (1 - SOC_range) * Qeff
Voc_range = Voc_func(q_range)
plt.subplot(1, 3, 2)
plt.plot(SOC_range, Voc_range)
plt.xlabel('SOC')
plt.ylabel('Voc (V)')
plt.title('SOC vs Voc')
plt.grid(True)

# 子图3: SOC vs 时间 (t)
plt.subplot(1, 3, 3)
plt.plot(sol.t, sol.y[0])
plt.xlabel('时间 (s)')
plt.ylabel('SOC')
plt.title('SOC vs 时间')
plt.grid(True)

plt.tight_layout()
plt.savefig('battery_curves.png')
print("图表已保存为 battery_curves.png")

# 可选：打印一些信息
print(f"放电时间: {sol.t[-1]:.2f} s")
print(f"最终 SOC: {sol.y[0][-1]:.4f}")
print(f"最终 Vrc: {sol.y[1][-1]:.4f}")
