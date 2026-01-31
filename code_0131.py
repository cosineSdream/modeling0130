import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ===============================
# 参数
# ===============================
p1 = 0.56777875        # V
p2 = -1.328625e-11     # V
lambda1 = -0.21950625  # Ah^-1
lambda2 = 2.330        # Ah^-1

R0 = 0.1032            # ohm
R1 = 0.0317            # ohm
C1 = 649.01            # F
Qeff = 5.0             # Ah
Ptotal = 10.0          # W

gamma = R1 * C1        # time constant (s)

# ===============================
# 初值
# ===============================
s0 = 1.0      # SOC
Vrc0 = 0.0    # 极化电压
x0 = [s0, Vrc0]

# ===============================
# ODE 定义
# ===============================
def odefun(t, x):
    s, Vrc = x

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
    events=stop_event
)

t = sol.t
s = sol.y[0]
Vrc = sol.y[1]

# ===============================
# 绘图：SOC vs time
# ===============================
plt.figure(figsize=(7, 5))
plt.plot(t / 60, s, linewidth=2)
plt.grid(True)
plt.xlabel("Time (min)")
plt.ylabel("SOC, s")
plt.title("SOC vs Time under Constant Power Load (P = 10 W)")
plt.show()
