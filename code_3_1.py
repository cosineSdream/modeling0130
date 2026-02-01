import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

try:
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK CN', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass

np.random.seed(42)
Time_limit = 86400*3  # 仿真总时长 (秒)
# ===============================
# ECM 参数表及三次多项式拟合
# ===============================
soc_table = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
R0_table  = np.array([0.0344, 0.0324, 0.0306, 0.0296, 0.0284, 0.0279, 0.0271, 0.0263, 0.0251])
R1_table  = np.array([0.0750, 0.0487, 0.0392, 0.0367, 0.0317, 0.0312, 0.0287, 0.0351, 0.0280])
C1_table  = np.array([427.09, 520.73, 567.06, 618.50, 649.01, 687.74, 720.09, 747.04, 769.39])

R0_coeffs = np.polyfit(soc_table, R0_table, 3)
R1_coeffs = np.polyfit(soc_table, R1_table, 3)
C1_coeffs = np.polyfit(soc_table, C1_table, 3)

R0_poly = np.poly1d(R0_coeffs)
R1_poly = np.poly1d(R1_coeffs)
C1_poly = np.poly1d(C1_coeffs)

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
print(f"R0 拟合优度 R² = {r_squared(R0_table, R0_fit):.6f}, RMSE = {rmse(R0_table, R0_fit)*1000:.4f} mΩ")
print(f"R1 拟合优度 R² = {r_squared(R1_table, R1_fit):.6f}, RMSE = {rmse(R1_table, R1_fit)*1000:.4f} mΩ")
print(f"C1 拟合优度 R² = {r_squared(C1_table, C1_fit):.6f}, RMSE = {rmse(C1_table, C1_fit):.4f} F")
print("=" * 70)


# ===============================================================
# 手机各子模块功率参数
# ===============================================================
bs       = 2.464  # 屏幕最大亮度功率 (W)
S_screen = 1   #归一化处理
Pcpumax  = 4.0
b_cpu    = 0.1

NET_PARAMS = {
    'WIFI6':    {'k': 288.5, 'idle': 11.5},
    '4G_LTE_A': {'k': 370.0, 'idle': 30.0},
    '5G_SA':    {'k': 510.0, 'idle': 40.0},
}

GPS_POWER = {
    'off':       0.0,
    'on':       49.2,
    'locating': 444.9,
}
def calc_P_screen(B):
    return bs * B * S_screen

def calc_P_cpu(u):
    return b_cpu + (Pcpumax - b_cpu) * u

def calc_P_net(net_state, r):
    p = NET_PARAMS[net_state]
    return (p['k'] * r + p['idle']) * 1e-3

def calc_P_gps(gps_state):
    return GPS_POWER[gps_state] * 1e-3


# ===============================================================
# 应用类型标记库（Marked Poisson Process 的标记空间）
# 每种应用类型定义了一套独立的功耗特征参数
# ===============================================================

# 应用分类定义
# 每个类型: label, 屏幕亮度B均值/标差, CPU负载u均值/标差,
#           传输速率r均值/标差, 网络状态, GPS状态,
#           持续时间均值(s)/标差(s) [LogNormal参数]
APP_TYPES = {
    'video': {
        'label': '视频播放',
        'B_mean': 0.72, 'B_std': 0.08,
        'u_mean': 0.22, 'u_std': 0.04,
        'r_mean': 0.70, 'r_std': 0.10,
        'net_state': 'WIFI6',
        'gps_state': 'off',
        'dur_mean': 480.0,   # 8 min
        'dur_std':  240.0,   # ~4 min std
    },
    'gaming': {
        'label': '玩游戏',
        'B_mean': 0.85, 'B_std': 0.05,
        'u_mean': 0.45, 'u_std': 0.03,
        'r_mean': 0.50, 'r_std': 0.06,
        'net_state': 'WIFI6',
        'gps_state': 'off',
        'dur_mean': 600.0,   # 10 min
        'dur_std':  300.0,
    },
    'social': {
        'label': '社交浏览',
        'B_mean': 0.60, 'B_std': 0.10,
        'u_mean': 0.15, 'u_std': 0.05,
        'r_mean': 0.45, 'r_std': 0.15,
        'net_state': 'WIFI6',
        'gps_state': 'off',
        'dur_mean': 300.0,   # 5 min
        'dur_std':  180.0,
    },
    'navigation': {
        'label': '导航',
        'B_mean': 0.60, 'B_std': 0.04,
        'u_mean': 0.20, 'u_std': 0.03,
        'r_mean': 0.10, 'r_std': 0.02,
        'net_state': '4G_LTE_A',
        'gps_state': 'locating',
        'dur_mean': 1200.0,  # 20 min
        'dur_std':  600.0,
    },
    'calling': {
        'label': '打电话',
        'B_mean': 0.0,  'B_std': 0.0,   # 屏幕关
        'u_mean': 0.10, 'u_std': 0.02,
        'r_mean': 0.10, 'r_std': 0.03,
        'net_state': '4G_LTE_A',
        'gps_state': 'off',
        'dur_mean': 180.0,   # 3 min
        'dur_std':   90.0,
    },
    'music': {
        'label': '听音乐(屏关)',
        'B_mean': 0.0,  'B_std': 0.0,
        'u_mean': 0.07, 'u_std': 0.02,
        'r_mean': 0.30, 'r_std': 0.08,
        'net_state': 'WIFI6',
        'gps_state': 'off',
        'dur_mean': 240.0,   # 4 min (一歌平均)
        'dur_std':   60.0,
    },
    'bg_download': {
        'label': '后台下载',
        'B_mean': 0.0,  'B_std': 0.0,   # 屏幕关
        'u_mean': 0.05, 'u_std': 0.01,
        'r_mean': 0.80, 'r_std': 0.10,
        'net_state': 'WIFI6',
        'gps_state': 'off',
        'dur_mean': 900.0,   # 15 min
        'dur_std':  450.0,
    },
    'bg_sync': {
        'label': '后台同步',
        'B_mean': 0.0,  'B_std': 0.0,
        'u_mean': 0.02, 'u_std': 0.005,
        'r_mean': 0.25, 'r_std': 0.08,
        'net_state': 'WIFI6',
        'gps_state': 'off',
        'dur_mean': 120.0,   # 2 min
        'dur_std':   60.0,
    },
}

# 应用类型离散概率分布
# 前台任务 vs 后台任务的比例也内含在此
APP_TYPE_KEYS = list(APP_TYPES.keys())
APP_TYPE_PROBS = np.array([
    0.22,   # video
    0.13,   # gaming
    0.20,   # social
    0.08,   # navigation
    0.07,   # calling
    0.10,   # music
    0.10,   # bg_download
    0.10,   # bg_sync
])
APP_TYPE_PROBS /= APP_TYPE_PROBS.sum()  # 归一化


# ===============================================================
# 昼夜使用强度函数 λ(t)  [events/hour]
# 模拟真实手机使用的昼夜节律
# ===============================================================
def lambda_intensity(t_sec):
    """
    返回时刻 t (秒) 对应的泊松到达强度 λ (events/hour).
    使用双高斯叠加模拟白天主峰+晚间副峰的使用习惯.
    """
    t_h = (t_sec % 86400) / 3600.0  # 转换为一天内的小时 [0, 24)

    # 白天主峰: 中午~下午 (中心 ~13h, 宽度 ~4h)
    peak1 = 8 * np.exp(-0.5 * ((t_h - 14.0) / 2)**2)
    # 晚间副峰: 晚上 (中心 ~21h, 宽度 ~3h)
    peak2 = 4 * np.exp(-0.5 * ((t_h - 21.0) / 2.5)**2)
    # 基础强度 (凌晨低水平)
    base  = 1

    return base + peak1 + peak2


# ===============================================================
# 基础待机功耗（屏幕关、极低CPU、网络idle）
# 始终存在，不依赖事件
# ===============================================================
def calc_P_base():
    """基础待机功耗: 屏幕关 + CPU空闲 + 网络WIFI6 idle + GPS off"""
    P_scr = calc_P_screen(0.0)
    P_cpu = calc_P_cpu(0.01)
    P_net = calc_P_net('WIFI6', 0.0)   # 仅 idle 功率
    P_gps = calc_P_gps('off')
    return P_scr + P_cpu + P_net + P_gps

P_BASE = calc_P_base()
print(f"\n基础待机功耗 P_base = {P_BASE*1000:.2f} mW\n")


# ===============================================================
# 单个事件的瞬时功耗贡献
# ===============================================================
def calc_event_power(event):
    """
    根据事件的采样参数计算其当前功耗贡献 (W).
    注意: 事件的 B, u, r 在开始时已采样并固定（简化处理），
    若要更精细可在此加入时间变化项.
    """
    app = APP_TYPES[event['app_type']]
    B   = event['B']
    u   = event['u']
    r   = event['r']

    # 对于屏幕关闭的事件，屏幕功耗为0（已在采样时处理）
    P_scr = calc_P_screen(B)
    # CPU增量（减去基础空闲部分，避免双重计数）
    P_cpu_inc = calc_P_cpu(u) - calc_P_cpu(0.02)
    # 网络增量（减去基础idle，避免双重计数）
    P_net_inc = calc_P_net(app['net_state'], r) - calc_P_net('WIFI6', 0.0)
    # GPS增量
    P_gps_inc = calc_P_gps(app['gps_state']) - calc_P_gps('off')

    return max(P_scr + P_cpu_inc + P_net_inc + P_gps_inc, 0.0)


# ===============================================================
# 采样单个事件（标记）
# ===============================================================
def sample_event(t_start):
    """
    在时刻 t_start 采样一个新事件.
    返回事件字典 (Marked Poisson 的一个标记 M_k).
    """
    # 离散采样: 应用类型 A_k
    app_idx = np.random.choice(len(APP_TYPE_KEYS), p=APP_TYPE_PROBS)
    app_type = APP_TYPE_KEYS[app_idx]
    app      = APP_TYPES[app_type]

    # 连续采样: 持续时间 D_k ~ LogNormal
    # LogNormal参数: 若 X~LogNormal(μ,σ²), 则 E[X]=exp(μ+σ²/2), Var[X]=(exp(σ²)-1)*exp(2μ+σ²)
    # 这里用均值/标差反解 μ, σ
    mean_d = app['dur_mean']
    std_d  = app['dur_std']
    sigma2 = np.log(1 + (std_d / mean_d)**2)
    mu     = np.log(mean_d) - sigma2 / 2
    D_k = np.random.lognormal(mean=mu, sigma=np.sqrt(sigma2))
    D_k = max(D_k, 10.0)  # 最短10秒

    # 连续采样: 功耗参数 Θ_k (带噪声的亮度/CPU/传输率)
    B_k = np.clip(np.random.normal(app['B_mean'], app['B_std']), 0.0, 1.0)
    u_k = np.clip(np.random.normal(app['u_mean'], app['u_std']), 0.02, 1.0)
    r_k = np.clip(np.random.normal(app['r_mean'], app['r_std']), 0.0, 1.0)

    return {
        'id':        id(object()),   # 唯一标识
        'app_type':  app_type,
        'label':     app['label'],
        't_start':   t_start,
        'duration':  D_k,
        't_end':     t_start + D_k,
        'B':         B_k,
        'u':         u_k,
        'r':         r_k,
    }


# ===============================================================
# OCV 参数
# ===============================================================
p1      = 0.56777875
p2      = -1.328625e-11
lambda1 = -0.21950625 * 2.29
lambda2 =  2.330 * 2.29
Qeff    = 5.0   # Ah


# ===============================================================
# Marked Poisson + ECM ODE 耦合仿真
# ===============================================================
def run_parallel_simulation(T_sim=Time_limit, dt=10.0):
    """
    核心仿真函数: 并行标记泊松过程 + Thevenin ECM 耦合.

    算法流程:
        1. 在离散时步 Δt 内检查是否有新事件到达 (泊松过程)
        2. 维护 ActiveSet: 移除已结束事件, 保留活跃事件
        3. 叠加所有活跃事件功耗 → 得到当前总负载 P(t)
        4. 用 RK4 积分 ECM ODE 推进一步

    Returns:
        dict: 包含完整时间序列和事件日志
    """
    # --- 时间序列初始化 ---
    n_steps = int(T_sim / dt)
    t_arr      = np.zeros(n_steps)
    soc_arr    = np.zeros(n_steps)
    Vt_arr     = np.zeros(n_steps)
    Voc_arr    = np.zeros(n_steps)
    IL_arr     = np.zeros(n_steps)
    Vrc_arr    = np.zeros(n_steps)
    P_total_arr= np.zeros(n_steps)
    P_base_arr = np.zeros(n_steps)
    n_active_arr = np.zeros(n_steps)

    # --- 状态初始化 ---
    soc = 1.0
    Vrc = 0.0

    # --- 事件管理 ---
    active_set  = []   # 当前活跃事件列表
    event_log   = []   # 全部事件记录（用于后续分析绘图）

    # --- 各类型累积时间记录 ---
    app_active_time = {k: 0.0 for k in APP_TYPES}

    # ---- RK4 积分一步 (对 [soc, Vrc] 给定 P_load) ----
    def rk4_step(soc_in, Vrc_in, P_load, h):
        def deriv(s, v, P):
            s_c  = np.clip(s, 0.05, 1.0)
            R0   = max(float(R0_poly(s_c)), 1e-6)
            R1   = max(float(R1_poly(s_c)), 1e-6)
            C1   = max(float(C1_poly(s_c)), 1e-6)
            gamma = R1 * C1

            q   = (1 - s) * Qeff
            Voc = p1*(np.exp(lambda1*q)-1) + p2*(np.exp(lambda2*q)-1) + 3.65

            Vt  = Voc - v
            disc = Vt**2 - 4*R0*P
            if disc < 0: disc = 0.0
            IL  = (Vt - np.sqrt(disc)) / (2*R0)

            dsdt   = -IL / Qeff / 3600.0
            dVrcdt = (IL*R1 - v) / gamma
            return dsdt, dVrcdt, IL, Voc, Vt

        k1_s, k1_v, _, _, _ = deriv(soc_in, Vrc_in, P_load)
        k2_s, k2_v, _, _, _ = deriv(soc_in + 0.5*h*k1_s, Vrc_in + 0.5*h*k1_v, P_load)
        k3_s, k3_v, _, _, _ = deriv(soc_in + 0.5*h*k2_s, Vrc_in + 0.5*h*k2_v, P_load)
        k4_s, k4_v, _, _, _ = deriv(soc_in + h*k3_s, Vrc_in + h*k3_v, P_load)

        soc_out = soc_in + (h/6)*(k1_s + 2*k2_s + 2*k3_s + k4_s)
        Vrc_out = Vrc_in + (h/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

        # 回算终点物理量
        _, _, IL_end, Voc_end, Vt_end = deriv(soc_out, Vrc_out, P_load)
        return soc_out, Vrc_out, IL_end, Voc_end, Vt_end

    # =========== 主仿真循环 ===========
    for i in range(n_steps):
        t = i * dt
        t_arr[i] = t

        # ── Step 1: 泊松到达检测 ──
        lam = lambda_intensity(t)         # events/hour
        p_arrive = lam * (dt / 3600.0)    # 在 Δt 内到达的概率
        # Poisson: 在短时间 Δt 内近似为 Bernoulli(p_arrive)
        # 若 p_arrive > 1 可能到达多个事件（用 Poisson 采样）
        if p_arrive <= 0.5:
            n_new = 1 if np.random.rand() < p_arrive else 0
        else:
            n_new = np.random.poisson(p_arrive)

        for _ in range(n_new):
            ev = sample_event(t)
            active_set.append(ev)
            event_log.append(ev)

        # ── Step 2: 清理已结束事件，累加功耗 ──
        still_active = []
        P_events = 0.0
        for ev in active_set:
            if t >= ev['t_end']:
                # 事件结束
                pass
            else:
                still_active.append(ev)
                P_events += calc_event_power(ev)
                app_active_time[ev['app_type']] += dt

        active_set = still_active

        # ── Step 3: 总负载 = 基础 + 事件叠加 ──
        P_load = P_BASE + P_events

        # ── Step 4: ECM ODE 积分 ──
        if soc <= 0.05:
            # SOC 耗尽，停止
            t_arr      = t_arr[:i]
            soc_arr    = soc_arr[:i]
            Vt_arr     = Vt_arr[:i]
            Voc_arr    = Voc_arr[:i]
            IL_arr     = IL_arr[:i]
            Vrc_arr    = Vrc_arr[:i]
            P_total_arr= P_total_arr[:i]
            P_base_arr = P_base_arr[:i]
            n_active_arr = n_active_arr[:i]
            break

        soc_new, Vrc_new, IL_val, Voc_val, Vt_val = rk4_step(soc, Vrc, P_load, dt)
        soc_new = max(soc_new, 0.05)

        # 记录
        soc_arr[i]      = soc
        Vrc_arr[i]      = Vrc
        Vt_arr[i]       = Vt_val
        Voc_arr[i]      = Voc_val
        IL_arr[i]       = IL_val
        P_total_arr[i]  = P_load
        P_base_arr[i]   = P_BASE
        n_active_arr[i] = len(active_set)

        # 推进状态
        soc = soc_new
        Vrc = Vrc_new

    discharge_time_h = t_arr[-1] / 3600.0 if len(t_arr) > 0 else 0.0

    return {
        't':             t_arr,
        'soc':           soc_arr,
        'Vt':            Vt_arr,
        'Voc':           Voc_arr,
        'IL':            IL_arr,
        'Vrc':           Vrc_arr,
        'P_total':       P_total_arr,
        'P_base':        P_base_arr,
        'n_active':      n_active_arr,
        'event_log':     event_log,
        'app_active_time': app_active_time,
        'discharge_time_h': discharge_time_h,
    }


# ===============================================================
# 运行仿真
# ===============================================================
print("=" * 70)
print("运行 Marked Poisson 并行任务 + ECM 耦合仿真 ...")
print("=" * 70)

sim = run_parallel_simulation(T_sim=Time_limit, dt=10.0)

print(f"\n放电时间: {sim['discharge_time_h']:.2f} h ({sim['discharge_time_h']*60:.1f} min)")
print(f"总事件数: {len(sim['event_log'])}")
print(f"\n各应用类型累积活跃时间:")
for k, v in sim['app_active_time'].items():
    print(f"  {APP_TYPES[k]['label']:<12}: {v/60:.1f} min")


# ===============================================================
# 颜色映射
# ===============================================================
APP_COLORS = {
    'video':       '#e74c3c',
    'gaming':      '#9b59b6',
    'social':      '#3498db',
    'navigation':  '#e67e22',
    'calling':     '#1abc9c',
    'music':       '#f39c12',
    'bg_download': '#2ecc71',
    'bg_sync':     '#95a5a6',
}


# ===============================================================
# 图 1: 泊松强度函数 λ(t) 昼夜节律
# ===============================================================
fig1, ax = plt.subplots(figsize=(12, 4))
t_24h = np.linspace(0, 86400, 1000)
lam_vals = [lambda_intensity(t) for t in t_24h]
ax.fill_between(t_24h/3600, lam_vals, alpha=0.3, color='#3498db')
ax.plot(t_24h/3600, lam_vals, linewidth=2.5, color='#2980b9')
ax.set_xlabel('时间 (h)', fontsize=12)
ax.set_ylabel('到达强度 λ (events/h)', fontsize=12)
ax.set_title('泊松过程到达强度函数 λ(t) — 昼夜使用节律', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 25, 2))
ax.grid(True, alpha=0.3)
ax.axvspan(0, 7, alpha=0.08, color='gray', label='深夜/凌晨')
ax.axvspan(22, 24, alpha=0.08, color='gray')
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()


# ===============================================================
# 图 2: 事件时间线（Gantt 图） + 功耗叠加曲线
# ===============================================================
fig2, (ax_gantt, ax_power) = plt.subplots(2, 1, figsize=(16, 9),
                                           gridspec_kw={'height_ratios': [2, 1]})
fig2.suptitle('并行事件时间线与总功耗演化', fontsize=15, fontweight='bold')

# --- Gantt: 事件时间线 ---
# 仅绘制在放电时间内的事件
t_end_sim = sim['t'][-1] if len(sim['t']) > 0 else Time_limit
events_in_range = [ev for ev in sim['event_log'] if ev['t_start'] < t_end_sim]

# 按应用类型分组绘制（y轴为类型索别）
type_keys_ordered = list(APP_TYPES.keys())
y_positions = {k: i for i, k in enumerate(type_keys_ordered)}

for ev in events_in_range:
    y    = y_positions[ev['app_type']]
    x0   = ev['t_start'] / 3600.0
    width= min(ev['duration'], t_end_sim - ev['t_start']) / 3600.0
    color= APP_COLORS[ev['app_type']]
    ax_gantt.barh(y, width, left=x0, height=0.7, color=color, alpha=0.75,
                  edgecolor='white', linewidth=0.4)

ax_gantt.set_yticks(list(y_positions.values()))
ax_gantt.set_yticklabels([APP_TYPES[k]['label'] for k in type_keys_ordered], fontsize=10)
ax_gantt.set_xlabel('时间 (h)', fontsize=11)
ax_gantt.set_title('各应用类型事件时间线（Gantt图）', fontsize=12)
ax_gantt.grid(axis='x', alpha=0.3)
ax_gantt.set_xlim(0, t_end_sim / 3600.0)

# 图例
legend_patches = [mpatches.Patch(color=APP_COLORS[k], label=APP_TYPES[k]['label'])
                  for k in type_keys_ordered]
ax_gantt.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)

# --- 功耗曲线: 总负载 + 活跃事件数 ---
t_h = sim['t'] / 3600.0
ax_power.plot(t_h, sim['P_total']*1000, linewidth=1.0, color='#e74c3c', alpha=0.7)
ax_power.fill_between(t_h, sim['P_base']*1000, sim['P_total']*1000,
                       alpha=0.25, color='#e74c3c', label='事件叠加功耗')
ax_power.axhline(y=sim['P_base'][0]*1000 if len(sim['P_base'])>0 else P_BASE*1000,
                 color='#2c3e50', linestyle='--', linewidth=1.5, label=f'基础待机 {P_BASE*1000:.1f} mW')
ax_power.set_xlabel('时间 (h)', fontsize=11)
ax_power.set_ylabel('总负载 P(t) (mW)', fontsize=11)
ax_power.set_title('总功耗随时间变化', fontsize=12)
ax_power.legend(fontsize=9, loc='upper left')
ax_power.grid(True, alpha=0.3)
ax_power.set_xlim(0, t_end_sim / 3600.0)

# 右轴: 活跃事件数
ax_n = ax_power.twinx()
ax_n.plot(t_h, sim['n_active'], linewidth=1.2, color='#27ae60', alpha=0.8)
ax_n.set_ylabel('活跃事件数', fontsize=11, color='#27ae60')
ax_n.tick_params(axis='y', labelcolor='#27ae60')
ax_n.set_ylim(0, max(sim['n_active'].max()+1, 6))

plt.tight_layout()
plt.show()


# ===============================================================
# 图 3: SOC / 端电压 / 放电电流 联合图
# ===============================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))
fig3.suptitle('电池状态演化（并行随机负载下）', fontsize=15, fontweight='bold')

# (0,0) SOC
ax = axes3[0, 0]
ax.plot(t_h, sim['soc'], linewidth=2, color='#2980b9')
ax.fill_between(t_h, 0.05, sim['soc'], alpha=0.1, color='#2980b9')
ax.axhline(y=0.05, color='red', linestyle=':', linewidth=1.5, label='截止 SOC=0.05')
ax.set_xlabel('时间 (h)', fontsize=11)
ax.set_ylabel('SOC', fontsize=11)
ax.set_title('SOC 随时间变化', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, t_end_sim / 3600.0)

# (0,1) 端电压 & OCV
ax = axes3[0, 1]
ax.plot(t_h, sim['Voc'], linewidth=1.8, color='#27ae60', label='Voc (开路电压)')
ax.plot(t_h, sim['Vt'],  linewidth=1.8, color='#e67e22', label='Vt (端电压)')
ax.plot(t_h, sim['Vrc'], linewidth=1.2, color='#9b59b6', linestyle='--', label='Vrc (极化电压)')
ax.set_xlabel('时间 (h)', fontsize=11)
ax.set_ylabel('电压 (V)', fontsize=11)
ax.set_title('电压随时间变化', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, t_end_sim / 3600.0)

# (1,0) 放电电流
ax = axes3[1, 0]
ax.plot(t_h, sim['IL'], linewidth=1.0, color='#e74c3c', alpha=0.8)
ax.fill_between(t_h, 0, sim['IL'], alpha=0.15, color='#e74c3c')
ax.set_xlabel('时间 (h)', fontsize=11)
ax.set_ylabel('放电电流 IL (A)', fontsize=11)
ax.set_title('放电电流随时间变化', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, t_end_sim / 3600.0)

# (1,1) 活跃事件数与总功耗双轴
ax = axes3[1, 1]
dt_plot = (sim['t'][1] - sim['t'][0]) if len(sim['t']) > 1 else 10.0
ax.bar(t_h, sim['n_active'], width=dt_plot/3600.0, color='#3498db', alpha=0.5, label='活跃事件数')
ax.set_xlabel('时间 (h)', fontsize=11)
ax.set_ylabel('活跃事件数', fontsize=11, color='#3498db')
ax.tick_params(axis='y', labelcolor='#3498db')
ax2 = ax.twinx()
ax2.plot(t_h, sim['P_total']*1000, linewidth=1.2, color='#e74c3c', label='总功耗')
ax2.set_ylabel('总功耗 (mW)', fontsize=11, color='#e74c3c')
ax2.tick_params(axis='y', labelcolor='#e74c3c')
ax.set_title('并行事件数与总功耗相关性', fontsize=12, fontweight='bold')
ax.set_xlim(0, t_end_sim / 3600.0)
# 合并图例
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()


# ===============================================================
# 图 4: 事件数量统计 & 各类型应用活跃时间饼图
# ===============================================================
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5.5))
fig4.suptitle('事件统计与时间分配分析', fontsize=14, fontweight='bold')

# (0) 各类型事件发生次数
type_counts = {k: 0 for k in APP_TYPES}
for ev in sim['event_log']:
    if ev['t_start'] < t_end_sim:
        type_counts[ev['app_type']] += 1

keys   = [k for k in APP_TYPES if type_counts[k] > 0]
counts = [type_counts[k] for k in keys]
colors = [APP_COLORS[k] for k in keys]
labels = [APP_TYPES[k]['label'] for k in keys]

axes4[0].bar(labels, counts, color=colors, edgecolor='#2c3e50', linewidth=0.7)
axes4[0].set_ylabel('事件次数', fontsize=11)
axes4[0].set_title('各应用类型触发次数', fontsize=12, fontweight='bold')
axes4[0].tick_params(axis='x', rotation=30)
axes4[0].grid(axis='y', alpha=0.3)
for i, c in enumerate(counts):
    axes4[0].text(i, c + 0.3, str(c), ha='center', va='bottom', fontsize=9, fontweight='bold')

# (1) 活跃时间饼图
active_times = {k: sim['app_active_time'][k] for k in APP_TYPES if sim['app_active_time'][k] > 0}
pie_keys   = list(active_times.keys())
pie_vals   = [active_times[k]/60 for k in pie_keys]  # 转分钟
pie_colors = [APP_COLORS[k] for k in pie_keys]
pie_labels = [f"{APP_TYPES[k]['label']}\n{active_times[k]/60:.0f}min" for k in pie_keys]

wedges, texts, autotexts = axes4[1].pie(
    pie_vals, colors=pie_colors, labels=pie_labels,
    autopct='%1.0f%%', startangle=90, pctdistance=0.82,
    textprops={'fontsize': 8}
)
for at in autotexts:
    at.set_fontsize(7.5)
    at.set_fontweight('bold')
axes4[1].set_title('各应用类型占用时间比例', fontsize=12, fontweight='bold')

# (2) 事件持续时间分布直方图
durations = [ev['duration']/60 for ev in sim['event_log'] if ev['t_start'] < t_end_sim]
axes4[2].hist(durations, bins=30, color='#8e44ad', edgecolor='white', linewidth=0.7, alpha=0.8)
axes4[2].set_xlabel('事件持续时间 (min)', fontsize=11)
axes4[2].set_ylabel('频次', fontsize=11)
axes4[2].set_title('事件持续时间分布 (LogNormal)', fontsize=12, fontweight='bold')
axes4[2].axvline(x=np.mean(durations), color='red', linestyle='--', linewidth=1.5,
                 label=f'均值 {np.mean(durations):.1f} min')
axes4[2].axvline(x=np.median(durations), color='orange', linestyle='--', linewidth=1.5,
                 label=f'中位数 {np.median(durations):.1f} min')
axes4[2].legend(fontsize=9)
axes4[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# ===============================================================
# 图 5: ECM 参数拟合曲线（保留原代码功能）
# ===============================================================
soc_fine = np.linspace(0.1, 0.9, 200)
fig5, axes5 = plt.subplots(1, 3, figsize=(15, 4.5))
fig5.suptitle('等效电路模型参数拟合 (三次多项式)', fontsize=14, fontweight='bold')

# R0
axes5[0].plot(soc_fine, R0_poly(soc_fine)*1000, linewidth=2, color='blue', label='Cubic Fit')
axes5[0].scatter(soc_table, R0_table*1000, color='red', s=60, zorder=5,
                 label='Data', edgecolor='black', linewidth=0.5)
axes5[0].set_xlabel('SOC', fontsize=11)
axes5[0].set_ylabel('R0 (mΩ)', fontsize=11)
axes5[0].set_title(f'R0 vs SOC  (R^2={r_squared(R0_table, R0_fit):.4f})', fontsize=11)
axes5[0].legend(fontsize=9); axes5[0].grid(True, alpha=0.3)

# R1
axes5[1].plot(soc_fine, R1_poly(soc_fine)*1000, linewidth=2, color='red', label='Cubic Fit')
axes5[1].scatter(soc_table, R1_table*1000, color='blue', s=60, zorder=5,
                 label='Data', edgecolor='black', linewidth=0.5)
axes5[1].set_xlabel('SOC', fontsize=11)
axes5[1].set_ylabel('R1 (mΩ)', fontsize=11)
axes5[1].set_title(f'R1 vs SOC  (R^2={r_squared(R1_table, R1_fit):.4f})', fontsize=11)
axes5[1].legend(fontsize=9); axes5[1].grid(True, alpha=0.3)

# C1
axes5[2].plot(soc_fine, C1_poly(soc_fine), linewidth=2, color='green', label='Cubic Fit')
axes5[2].scatter(soc_table, C1_table, color='orange', s=60, zorder=5,
                 label='Data', edgecolor='black', linewidth=0.5)
axes5[2].set_xlabel('SOC', fontsize=11)
axes5[2].set_ylabel('C1 (F)', fontsize=11)
axes5[2].set_title(f'C1 vs SOC  (R^2={r_squared(C1_table, C1_fit):.4f})', fontsize=11)
axes5[2].legend(fontsize=9); axes5[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ===============================================================
# 图 6: 网络功率对比（保留原代码）
# ===============================================================
r_sweep = np.linspace(0, 1, 200)
fig6, ax = plt.subplots(figsize=(9, 5))
for net_key, color in [('WIFI6','#3498db'), ('4G_LTE_A','#e67e22'), ('5G_SA','#e74c3c')]:
    p = NET_PARAMS[net_key]
    P_net_sweep = (p['k'] * r_sweep + p['idle'])
    ax.plot(r_sweep, P_net_sweep, linewidth=2.5, color=color, label=net_key.replace('_',' '))
ax.set_xlabel('归一化传输速率 r(t)', fontsize=12)
ax.set_ylabel('网络功率 (mW)', fontsize=12)
ax.set_title('网络模块功率 vs 传输速率', fontsize=13, fontweight='bold')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ===============================================================
# 最终汇总输出
# ===============================================================
print("\n" + "=" * 70)
print("仿真汇总")
print("=" * 70)
print(f"  仿真时长:          {Time_limit/3600:.0f} h")
print(f"  实际放电时间:      {sim['discharge_time_h']:.2f} h ({sim['discharge_time_h']*60:.1f} min)")
print(f"  基础待机功耗:      {P_BASE*1000:.2f} mW")
print(f"  平均总负载:        {np.mean(sim['P_total'])*1000:.2f} mW")
print(f"  最大总负载:        {np.max(sim['P_total'])*1000:.2f} mW")
print(f"  最大并行事件数:    {int(np.max(sim['n_active']))}")
print(f"  总事件触发数:      {len(sim['event_log'])}")
print("=" * 70)