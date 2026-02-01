import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import matplotlib
# 尝试启用中文字体渲染
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass

# ===============================
# 从表格中提取的数据
# ===============================
soc_table = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
R0_table = np.array([0.0344, 0.0324, 0.0306, 0.0296, 0.0284, 0.0279, 0.0271, 0.0263, 0.0251])
R1_table = np.array([0.0750, 0.0487, 0.0392, 0.0367, 0.0317, 0.0312, 0.0287, 0.0351, 0.0280])
C1_table = np.array([427.09, 520.73, 567.06, 618.50, 649.01, 687.74, 720.09, 747.04, 769.39])

# ===============================
# 三次多项式拟合
# ===============================
R0_coeffs = np.polyfit(soc_table, R0_table, 3)
R1_coeffs = np.polyfit(soc_table, R1_table, 3)
C1_coeffs = np.polyfit(soc_table, C1_table, 3)

R0_poly = np.poly1d(R0_coeffs)
R1_poly = np.poly1d(R1_coeffs)
C1_poly = np.poly1d(C1_coeffs)

# 拟合优度
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

R0_fit = R0_poly(soc_table)
R1_fit = R1_poly(soc_table)
C1_fit = C1_poly(soc_table)

print("=" * 70)
print("三次多项式拟合结果")
print("=" * 70)
print(f"R0(s) = {R0_coeffs[0]:.8f}*s^3 + {R0_coeffs[1]:.8f}*s^2 + {R0_coeffs[2]:.8f}*s + {R0_coeffs[3]:.8f}")
print(f"R1(s) = {R1_coeffs[0]:.8f}*s^3 + {R1_coeffs[1]:.8f}*s^2 + {R1_coeffs[2]:.8f}*s + {R1_coeffs[3]:.8f}")
print(f"C1(s) = {C1_coeffs[0]:.6f}*s^3 + {C1_coeffs[1]:.6f}*s^2 + {C1_coeffs[2]:.6f}*s + {C1_coeffs[3]:.6f}")
print("=" * 70)
print(f"\nR0 拟合优度 R² = {r_squared(R0_table, R0_fit):.6f}, RMSE = {rmse(R0_table, R0_fit)*1000:.4f} mΩ")
print(f"R1 拟合优度 R² = {r_squared(R1_table, R1_fit):.6f}, RMSE = {rmse(R1_table, R1_fit)*1000:.4f} mΩ")
print(f"C1 拟合优度 R² = {r_squared(C1_table, C1_fit):.6f}, RMSE = {rmse(C1_table, C1_fit):.4f} F")
print("=" * 70)


# ===============================================================
# ========  智能手机各模块功率模型参数  ========
# ===============================================================

# ---- 1. 屏幕功率参数 ----
# bs: 屏幕满亮度时单位面积功率 (W/m²)
# 典型手机屏幕: AMOLED, 约 6 寸, 亮度峰值对应约 350 W/m²
bs = 350.0          # W/m²
# S: 屏幕面积 (m²)
# 6.1寸 对角线屏幕, 宽高比约 19.5:9 => 约 0.0076 m²
S_screen = 0.0076   # m²

# ---- 2. CPU功率参数 ----
Pcpumax = 6.0       # W, CPU 满负载功率
b_cpu   = 0.3       # W, CPU 空闲功率 (负载为零时的静态功率)

# ---- 3. 网络模块功率参数 (单位 mW) ----
# Pnet = k * r(t) + P_idle
# WIFI6:    Pnet = 288.5 * r + 11.5   (mW)
# 4G LTE-A: Pnet = 370   * r + 30     (mW)
# 5G SA:    Pnet = 510   * r + 40     (mW)
NET_PARAMS = {
    'WIFI6':    {'k': 288.5, 'idle': 11.5},
    '4G_LTE_A': {'k': 370.0, 'idle': 30.0},
    '5G_SA':    {'k': 510.0, 'idle': 40.0},
}

# ---- 4. GPS功率参数 (mW) ----
GPS_POWER = {
    'off':       0.0,
    'on':       49.2,
    'locating': 444.9,
}

# ===============================================================
# ========  使用场景预设  ========
# ===============================================================
# 每种场景定义:
#   B          : 屏幕归一化亮度 [0,1]
#   u          : CPU归一化负载 [0,1]
#   r          : 归一化传输速率 [0,1]
#   net_state  : 网络状态 key
#   gps_state  : GPS状态 key
#   duration   : 持续时间 (s), -1 表示一直到电池耗尽
#   label      : 显示标签

SCENARIOS = {
    'gaming': {
        'label': '玩游戏',
        'B': 0.85, 'u': 0.90, 'r': 0.60,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'duration': -1,
    },
    'music': {
        'label': '听音乐(屏关)',
        'B': 0.0,  'u': 0.15, 'r': 0.30,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'duration': -1,
    },
    'calling': {
        'label': '打电话',
        'B': 0.0,  'u': 0.20, 'r': 0.10,
        'net_state': '4G_LTE_A', 'gps_state': 'off',
        'duration': -1,
    },
    'navigation': {
        'label': '导航',
        'B': 0.90, 'u': 0.50, 'r': 0.40,
        'net_state': '4G_LTE_A', 'gps_state': 'locating',
        'duration': -1,
    },
    'video': {
        'label': '看视频',
        'B': 0.70, 'u': 0.45, 'r': 0.70,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'duration': -1,
    },
    'standby': {
        'label': '待机(屏关)',
        'B': 0.0,  'u': 0.02, 'r': 0.05,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'duration': -1,
    },
    'mixed_light': {
        'label': '轻度混合使用',
        'B': 0.50, 'u': 0.25, 'r': 0.30,
        'net_state': 'WIFI6', 'gps_state': 'on',
        'duration': -1,
    },
    'mixed_heavy': {
        'label': '重度混合使用',
        'B': 0.80, 'u': 0.70, 'r': 0.55,
        'net_state': '5G_SA',  'gps_state': 'locating',
        'duration': -1,
    },
}


# ===============================================================
# ========  手机各子模块功率计算函数  ========
# ===============================================================

def calc_P_screen(B):
    """屏幕功率 (W): P_screen = bs * B(t) * S"""
    return bs * B * S_screen

def calc_P_cpu(u):
    """
    CPU功率 (W):
        P_cpu = b_cpu + (Pcpumax - b_cpu) * u(t)
    当 u=0 时 P_cpu = b_cpu (空闲功率);
    当 u=1 时 P_cpu = Pcpumax (满负载功率).
    """
    return b_cpu + (Pcpumax - b_cpu) * u

def calc_P_net(net_state, r):
    """网络功率 (W): 由 mW 转换为 W"""
    p = NET_PARAMS[net_state]
    return (p['k'] * r + p['idle']) * 1e-3   # mW -> W

def calc_P_gps(gps_state):
    """GPS功率 (W): 由 mW 转换为 W"""
    return GPS_POWER[gps_state] * 1e-3       # mW -> W

def calc_P_total_phone(scenario_key):
    """给定场景key，计算手机总负载功率 P_total (W)"""
    sc = SCENARIOS[scenario_key]
    P_scr = calc_P_screen(sc['B'])
    P_cpu = calc_P_cpu(sc['u'])
    P_net = calc_P_net(sc['net_state'], sc['r'])
    P_gps = calc_P_gps(sc['gps_state'])
    return P_scr + P_cpu + P_net + P_gps, P_scr, P_cpu, P_net, P_gps


# ===============================================================
# ========  OCV 参数  ========
# ===============================================================
p1       = 0.56777875
p2       = -1.328625e-11
lambda1  = -0.21950625 * 2.29   # Ah^-1
lambda2   = 2.330 * 2.29        # Ah^-1

Qeff = 5.0   # Ah (电池容量)


# ===============================================================
# ========  ODE 定义 (恒功率放电)  ========
# ===============================================================
def odefun(t, x, P_load):
    """
    状态向量: x = [SOC, Vrc]
    P_load: 恒功率负载 (W)
    """
    s, Vrc = x
    s_clipped = np.clip(s, 0.05, 1.0)

    R0 = max(float(R0_poly(s_clipped)), 1e-6)
    R1 = max(float(R1_poly(s_clipped)), 1e-6)
    C1 = max(float(C1_poly(s_clipped)), 1e-6)
    gamma = R1 * C1

    # Voc
    q   = (1 - s) * Qeff
    Voc = p1 * (np.exp(lambda1 * q) - 1) \
        + p2 * (np.exp(lambda2 * q) - 1) \
        + 3.65

    # 恒功率 => 二次方程求 IL
    Vt  = Voc - Vrc
    disc = Vt**2 - 4 * R0 * P_load
    if disc < 0:
        disc = 0.0
    IL = (Vt - np.sqrt(disc)) / (2 * R0)

    dsdt   = -IL / Qeff / 3600.0
    dVrcdt = (IL * R1 - Vrc) / gamma
    return [dsdt, dVrcdt]


# ===============================================================
# ========  事件: SOC 降至 0.05 停止  ========
# ===============================================================
def stop_event(t, x, P_load):
    return x[0] - 0.05

stop_event.terminal   = True
stop_event.direction  = -1


# ===============================================================
# ========  单场景求解并返回结果字典  ========
# ===============================================================
def run_scenario(scenario_key):
    P_total, P_scr, P_cpu, P_net, P_gps = calc_P_total_phone(scenario_key)

    x0     = [1.0, 0.0]                       # SOC=1, Vrc=0
    t_span = (0, 24 * 3600)                   # 最大24小时

    sol = solve_ivp(
        odefun, t_span, x0,
        args=(P_total,),
        method='RK45', rtol=1e-8, atol=1e-10,
        events=stop_event, dense_output=True
    )

    t   = sol.t
    soc = sol.y[0]
    Vrc = sol.y[1]

    # 逐点回算物理量
    IL_arr, Voc_arr, Vt_arr = [], [], []
    for i in range(len(t)):
        s_c  = np.clip(soc[i], 0.05, 1.0)
        R0   = max(float(R0_poly(s_c)), 1e-6)
        q    = (1 - soc[i]) * Qeff
        Voc  = p1*(np.exp(lambda1*q)-1) + p2*(np.exp(lambda2*q)-1) + 3.65
        Vt_v = Voc - Vrc[i]
        disc = Vt_v**2 - 4*R0*P_total
        if disc < 0: disc = 0.0
        IL   = (Vt_v - np.sqrt(disc)) / (2*R0)
        IL_arr.append(IL); Voc_arr.append(Voc); Vt_arr.append(Vt_v)

    IL_arr  = np.array(IL_arr)
    Voc_arr = np.array(Voc_arr)
    Vt_arr  = np.array(Vt_arr)
    power_arr = IL_arr * Vt_arr
    energy_cum = np.cumsum(power_arr * np.diff(np.concatenate(([0], t)))) / 3600  # Wh

    return {
        'key': scenario_key,
        'label': SCENARIOS[scenario_key]['label'],
        'P_total': P_total, 'P_scr': P_scr, 'P_cpu': P_cpu,
        'P_net': P_net, 'P_gps': P_gps,
        't': t, 'soc': soc, 'Vrc': Vrc,
        'IL': IL_arr, 'Voc': Voc_arr, 'Vt': Vt_arr,
        'power': power_arr, 'energy_cum': energy_cum,
        'discharge_time_min': t[-1] / 60,
        'discharge_time_h':   t[-1] / 3600,
        'energy_total_Wh':    energy_cum[-1],
    }


# ===============================================================
# ========  运行全部场景  ========
# ===============================================================
print("\n" + "=" * 70)
print("运行各使用场景仿真 ...")
print("=" * 70)

results = {}
for key in SCENARIOS:
    results[key] = run_scenario(key)
    r = results[key]
    print(f"\n[{r['label']}]  总负载 P = {r['P_total']*1000:.1f} mW"
          f"  (屏幕 {r['P_scr']*1000:.1f}, CPU {r['P_cpu']*1000:.1f},"
          f" 网络 {r['P_net']*1000:.1f}, GPS {r['P_gps']*1000:.1f} mW)")
    print(f"  放电时间: {r['discharge_time_h']:.2f} h ({r['discharge_time_min']:.1f} min),"
          f"  总输出能量: {r['energy_total_Wh']:.3f} Wh")


# ===============================================================
# ========  绘图 ========
# ===============================================================
# 颜色映射
COLORS = {
    'gaming':       '#e74c3c',
    'music':        '#9b59b6',
    'calling':      '#3498db',
    'navigation':   '#e67e22',
    'video':        '#1abc9c',
    'standby':      '#95a5a6',
    'mixed_light':  '#2ecc71',
    'mixed_heavy':  '#c0392b',
}

# ─────────────────────────────────────────────
# 图 1 : 功率分解柱状图 (各场景)
# ─────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(14, 6))

labels   = [results[k]['label'] for k in SCENARIOS]
P_scr_v  = [results[k]['P_scr']*1000  for k in SCENARIOS]
P_cpu_v  = [results[k]['P_cpu']*1000  for k in SCENARIOS]
P_net_v  = [results[k]['P_net']*1000  for k in SCENARIOS]
P_gps_v  = [results[k]['P_gps']*1000  for k in SCENARIOS]

x_pos = np.arange(len(labels))
w     = 0.55

b1 = ax.bar(x_pos, P_scr_v,                          width=w, color='#f39c12', label='屏幕 P_screen')
b2 = ax.bar(x_pos, P_cpu_v, bottom=P_scr_v,          width=w, color='#e74c3c', label='CPU  P_cpu')
b3 = ax.bar(x_pos, P_net_v,
            bottom=[a+b for a,b in zip(P_scr_v, P_cpu_v)], width=w,
            color='#3498db', label='网络 P_net')
b4 = ax.bar(x_pos, P_gps_v,
            bottom=[a+b+c for a,b,c in zip(P_scr_v, P_cpu_v, P_net_v)], width=w,
            color='#2ecc71', label='GPS  P_gps')

# 总功率标注
for i, k in enumerate(SCENARIOS):
    total = results[k]['P_total']*1000
    ax.text(i, total + 30, f'{total:.0f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#2c3e50')

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=10, rotation=15)
ax.set_ylabel('功率 (mW)', fontsize=12)
ax.set_title('各使用场景功率分解', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max([results[k]['P_total'] for k in SCENARIOS])*1000 * 1.15)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 图 2 : SOC vs 时间 (所有场景对比)
# ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))

# --- 子图 (0,0): SOC vs Time ---
ax = axes2[0, 0]
for k in SCENARIOS:
    r = results[k]
    ax.plot(r['t']/3600, r['soc'], linewidth=2, color=COLORS[k], label=r['label'])
ax.set_xlabel('时间 (h)', fontsize=11)
ax.set_ylabel('SOC', fontsize=11)
ax.set_title('各场景 SOC 随时间变化', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='best', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

# --- 子图 (0,1): 放电时间柱状图 ---
ax = axes2[0, 1]
discharge_h = [results[k]['discharge_time_h'] for k in SCENARIOS]
colors_list = [COLORS[k] for k in SCENARIOS]
bars = ax.barh(labels, discharge_h, color=colors_list, edgecolor='#2c3e50', linewidth=0.8)
for bar, val in zip(bars, discharge_h):
    ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
            f'{val:.2f} h', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('放电时间 (h)', fontsize=11)
ax.set_title('各场景使用寿命', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, max(discharge_h)*1.25)

# --- 子图 (1,0): 端电压 Vt vs Time (选几个典型场景) ---
ax = axes2[1, 0]
highlight = ['gaming', 'navigation', 'video', 'mixed_light', 'standby']
for k in highlight:
    r = results[k]
    ax.plot(r['t']/3600, r['Vt'], linewidth=2, color=COLORS[k], label=r['label'])
ax.set_xlabel('时间 (h)', fontsize=11)
ax.set_ylabel('端电压 Vt (V)', fontsize=11)
ax.set_title('典型场景端电压随时间变化', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- 子图 (1,1): 累计输出能量 ---
ax = axes2[1, 1]
for k in SCENARIOS:
    r = results[k]
    ax.plot(r['t']/3600, r['energy_cum'], linewidth=2, color=COLORS[k], label=r['label'])
ax.set_xlabel('时间 (h)', fontsize=11)
ax.set_ylabel('累计能量输出 (Wh)', fontsize=11)
ax.set_title('各场景累计能量输出', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='best', ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 图 3 : 单场景详细仿真图 (以"玩游戏"为例，保留原代码风格)
# ─────────────────────────────────────────────
# 选择一个典型场景做详细展示
DETAIL_SCENARIO = 'gaming'
r = results[DETAIL_SCENARIO]
P_load = r['P_total']

# 回算各ECM参数数组
t  = r['t']
soc_arr = r['soc']
Vrc_arr = r['Vrc']

R0_arr, R1_arr, C1_arr, gamma_arr = [], [], [], []
for i in range(len(t)):
    s_c  = np.clip(soc_arr[i], 0.05, 1.0)
    R0v  = max(float(R0_poly(s_c)), 1e-6)
    R1v  = max(float(R1_poly(s_c)), 1e-6)
    C1v  = max(float(C1_poly(s_c)), 1e-6)
    R0_arr.append(R0v); R1_arr.append(R1v)
    C1_arr.append(C1v); gamma_arr.append(R1v*C1v)

R0_arr   = np.array(R0_arr)
R1_arr   = np.array(R1_arr)
C1_arr   = np.array(C1_arr)
gamma_arr= np.array(gamma_arr)

fig3 = plt.figure(figsize=(18, 12))
suptitle = f'详细仿真: {r["label"]}  (总负载 P = {P_load*1000:.1f} mW)'
fig3.suptitle(suptitle, fontsize=15, fontweight='bold', y=1.01)

# 4×3 子图
ax1 = plt.subplot(4, 3, 1)
ax1.plot(t/60, soc_arr, linewidth=2, color='b')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel("Time (min)", fontsize=10)
ax1.set_ylabel("SOC", fontsize=10)
ax1.set_title("SOC vs Time", fontsize=11)

ax2 = plt.subplot(4, 3, 2)
ax2.plot(t/60, r['IL'], linewidth=2, color='r')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel("Time (min)", fontsize=10)
ax2.set_ylabel("Current (A)", fontsize=10)
ax2.set_title("Discharge Current vs Time", fontsize=11)

ax3 = plt.subplot(4, 3, 3)
ax3.plot(t/60, r['Voc'], linewidth=2, label='Voc', color='g')
ax3.plot(t/60, r['Vt'],  linewidth=2, label='Vt (Terminal)', color='orange')
ax3.plot(t/60, Vrc_arr,  linewidth=2, label='Vrc (Polarization)', color='purple')
ax3.grid(True, alpha=0.3)
ax3.set_xlabel("Time (min)", fontsize=10)
ax3.set_ylabel("Voltage (V)", fontsize=10)
ax3.set_title("Voltages vs Time", fontsize=11)
ax3.legend(fontsize=8)

ax4 = plt.subplot(4, 3, 4)
ax4.plot(t/60, R0_arr*1000, linewidth=2, color='darkblue')
ax4.grid(True, alpha=0.3)
ax4.set_xlabel("Time (min)", fontsize=10)
ax4.set_ylabel("R0 (mΩ)", fontsize=10)
ax4.set_title("R0 vs Time", fontsize=11)

ax5 = plt.subplot(4, 3, 5)
ax5.plot(t/60, R1_arr*1000, linewidth=2, color='darkred')
ax5.grid(True, alpha=0.3)
ax5.set_xlabel("Time (min)", fontsize=10)
ax5.set_ylabel("R1 (mΩ)", fontsize=10)
ax5.set_title("R1 vs Time", fontsize=11)

ax6 = plt.subplot(4, 3, 6)
ax6.plot(t/60, C1_arr, linewidth=2, color='darkgreen')
ax6.grid(True, alpha=0.3)
ax6.set_xlabel("Time (min)", fontsize=10)
ax6.set_ylabel("C1 (F)", fontsize=10)
ax6.set_title("C1 vs Time", fontsize=11)

# 第三行: 拟合曲线 vs 原始数据
soc_fine = np.linspace(0.1, 0.9, 200)

ax7 = plt.subplot(4, 3, 7)
ax7.plot(soc_fine, R0_poly(soc_fine)*1000, linewidth=2, color='blue', label='Cubic Fit')
ax7.scatter(soc_table, R0_table*1000, color='red', s=60, zorder=5,
            label='Data', edgecolor='black', linewidth=0.5)
ax7.grid(True, alpha=0.3)
ax7.set_xlabel("SOC", fontsize=10)
ax7.set_ylabel("R0 (mΩ)", fontsize=10)
ax7.set_title(f"R0 vs SOC (R²={r_squared(R0_table, R0_fit):.4f})", fontsize=11)
ax7.legend(fontsize=8)

ax8 = plt.subplot(4, 3, 8)
ax8.plot(soc_fine, R1_poly(soc_fine)*1000, linewidth=2, color='red', label='Cubic Fit')
ax8.scatter(soc_table, R1_table*1000, color='blue', s=60, zorder=5,
            label='Data', edgecolor='black', linewidth=0.5)
ax8.grid(True, alpha=0.3)
ax8.set_xlabel("SOC", fontsize=10)
ax8.set_ylabel("R1 (mΩ)", fontsize=10)
ax8.set_title(f"R1 vs SOC (R²={r_squared(R1_table, R1_fit):.4f})", fontsize=11)
ax8.legend(fontsize=8)

ax9 = plt.subplot(4, 3, 9)
ax9.plot(soc_fine, C1_poly(soc_fine), linewidth=2, color='green', label='Cubic Fit')
ax9.scatter(soc_table, C1_table, color='orange', s=60, zorder=5,
            label='Data', edgecolor='black', linewidth=0.5)
ax9.grid(True, alpha=0.3)
ax9.set_xlabel("SOC", fontsize=10)
ax9.set_ylabel("C1 (F)", fontsize=10)
ax9.set_title(f"C1 vs SOC (R²={r_squared(C1_table, C1_fit):.4f})", fontsize=11)
ax9.legend(fontsize=8)

# 第四行: 时间常数, 功率, 累计能量
ax10 = plt.subplot(4, 3, 10)
ax10.plot(t/60, gamma_arr, linewidth=2, color='darkviolet')
ax10.grid(True, alpha=0.3)
ax10.set_xlabel("Time (min)", fontsize=10)
ax10.set_ylabel("γ = R1·C1 (s)", fontsize=10)
ax10.set_title("Time Constant vs Time", fontsize=11)

ax11 = plt.subplot(4, 3, 11)
ax11.plot(t/60, r['power'], linewidth=2, color='darkorange')
ax11.axhline(y=P_load, color='k', linestyle='--', linewidth=1, label=f'Target: {P_load*1000:.1f} mW')
ax11.grid(True, alpha=0.3)
ax11.set_xlabel("Time (min)", fontsize=10)
ax11.set_ylabel("Power (W)", fontsize=10)
ax11.set_title("Output Power vs Time", fontsize=11)
ax11.legend(fontsize=8)

ax12 = plt.subplot(4, 3, 12)
ax12.plot(t/60, r['energy_cum'], linewidth=2, color='brown')
ax12.grid(True, alpha=0.3)
ax12.set_xlabel("Time (min)", fontsize=10)
ax12.set_ylabel("Cumulative Energy (Wh)", fontsize=10)
ax12.set_title("Cumulative Energy Output vs Time", fontsize=11)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 图 4 : Voc vs q 和 Voc vs SOC (原有图保留)
# ─────────────────────────────────────────────
q_arr = (1 - soc_arr) * Qeff

fig4, (ax_q, ax_s) = plt.subplots(1, 2, figsize=(12, 5))

ax_q.plot(q_arr, r['Voc'], linewidth=2, color='teal')
ax_q.grid(True, alpha=0.3)
ax_q.set_xlabel("Discharged Capacity q (Ah)", fontsize=11)
ax_q.set_ylabel("Open Circuit Voltage Voc (V)", fontsize=11)
ax_q.set_title("Voc vs Discharged Capacity", fontsize=12)

ax_s.plot(soc_arr, r['Voc'], linewidth=2, color='darkorange')
ax_s.grid(True, alpha=0.3)
ax_s.set_xlabel("State of Charge (SOC)", fontsize=11)
ax_s.set_ylabel("Open Circuit Voltage Voc (V)", fontsize=11)
ax_s.set_title("Voc vs SOC", fontsize=12)
ax_s.invert_xaxis()

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 图 5 : 网络功率对比 (r 从 0~1 扫描)
# ─────────────────────────────────────────────
r_sweep = np.linspace(0, 1, 200)
fig5, ax = plt.subplots(figsize=(9, 5))
for net_key, color in [('WIFI6','#3498db'), ('4G_LTE_A','#e67e22'), ('5G_SA','#e74c3c')]:
    p = NET_PARAMS[net_key]
    P_net_sweep = (p['k'] * r_sweep + p['idle'])   # mW
    ax.plot(r_sweep, P_net_sweep, linewidth=2.5, color=color, label=net_key.replace('_',' '))
ax.set_xlabel('归一化传输速率 r(t)', fontsize=12)
ax.set_ylabel('网络功率 (mW)', fontsize=12)
ax.set_title('网络模块功率 vs 传输速率', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 最终统计输出
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("各场景仿真汇总")
print("=" * 70)
print(f"{'场景':<14} {'总功率(mW)':>10} {'屏幕':>8} {'CPU':>8} {'网络':>8} {'GPS':>8} {'使用时间(h)':>12} {'能量(Wh)':>10}")
print("-" * 70)
for k in SCENARIOS:
    rv = results[k]
    print(f"{rv['label']:<14} "
          f"{rv['P_total']*1000:>9.1f} "
          f"{rv['P_scr']*1000:>7.1f} "
          f"{rv['P_cpu']*1000:>7.1f} "
          f"{rv['P_net']*1000:>7.1f} "
          f"{rv['P_gps']*1000:>7.1f} "
          f"{rv['discharge_time_h']:>11.2f} "
          f"{rv['energy_total_Wh']:>9.3f}")
print("=" * 70)