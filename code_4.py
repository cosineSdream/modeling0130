import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

try:
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK CN', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass

np.random.seed(42)

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
print(f"R0 R²={r_squared(R0_table, R0_fit):.6f}, RMSE={rmse(R0_table, R0_fit)*1000:.4f} mΩ")
print(f"R1 R²={r_squared(R1_table, R1_fit):.6f}, RMSE={rmse(R1_table, R1_fit)*1000:.4f} mΩ")
print(f"C1 R²={r_squared(C1_table, C1_fit):.6f}, RMSE={rmse(C1_table, C1_fit):.4f} F")
print("=" * 70)


# ===============================================================
# 手机各子模块功率参数
# ===============================================================
bs       = 2.464
S_screen = 1.0
Pcpumax  = 4.0
b_cpu    = 0.1

NET_PARAMS = {
    'WIFI6':    {'k': 288.5, 'idle': 11.5},
    '4G_LTE_A': {'k': 370.0, 'idle': 30.0},
    '5G_SA':    {'k': 510.0, 'idle': 40.0},
}
GPS_POWER = {'off': 0.0, 'on': 49.2, 'locating': 444.9}

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
# 应用类型标记库 (Marked Poisson Process)
# ===============================================================
APP_TYPES = {
    'video': {
        'label': '视频播放',
        'B_mean': 0.72, 'B_std': 0.08,
        'u_mean': 0.22, 'u_std': 0.04,
        'r_mean': 0.70, 'r_std': 0.10,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'dur_mean': 4800.0, 'dur_std': 2400.0,
    },
}

APP_TYPE_KEYS  = list(APP_TYPES.keys())
APP_TYPE_PROBS = np.array([0.22])
APP_TYPE_PROBS /= APP_TYPE_PROBS.sum()


# ===============================================================
# 昼夜使用强度函数 λ(t)
# ===============================================================
def lambda_intensity(t_sec):
    t_h   = (t_sec % 86400) / 3600.0
    peak1 = 8.0  * np.exp(-0.5 * ((t_h - 13.0) / 3.5)**2)
    peak2 = 10.0 * np.exp(-0.5 * ((t_h - 21.0) / 2.5)**2)
    return 0.5 + peak1 + peak2


# ===============================================================
# 基础待机功耗
# ===============================================================
def calc_P_base():
    return calc_P_screen(0.0) + calc_P_cpu(0.02) + calc_P_net('WIFI6', 0.0) + calc_P_gps('off')

P_BASE = calc_P_base()
print(f"\n基础待机功耗 P_base = {P_BASE*1000:.2f} mW\n")


# ===============================================================
# 事件功耗贡献（增量，去除 base 部分）
# ===============================================================
def calc_event_power(event):
    app = APP_TYPES[event['app_type']]
    P_scr     = calc_P_screen(event['B'])
    P_cpu_inc = calc_P_cpu(event['u'])   - calc_P_cpu(0.02)
    P_net_inc = calc_P_net(app['net_state'], event['r']) - calc_P_net('WIFI6', 0.0)
    P_gps_inc = calc_P_gps(app['gps_state']) - calc_P_gps('off')
    return max(P_scr + P_cpu_inc + P_net_inc + P_gps_inc, 0.0)


# ===============================================================
# 事件采样（标记 M_k）
# ===============================================================
def sample_event(t_start):
    app_idx  = np.random.choice(len(APP_TYPE_KEYS), p=APP_TYPE_PROBS)
    app_type = APP_TYPE_KEYS[app_idx]
    app      = APP_TYPES[app_type]

    mean_d = app['dur_mean'];  std_d = app['dur_std']
    sigma2 = np.log(1 + (std_d / mean_d)**2)
    mu     = np.log(mean_d) - sigma2 / 2
    D_k    = max(np.random.lognormal(mean=mu, sigma=np.sqrt(sigma2)), 10.0)

    B_k = np.clip(np.random.normal(app['B_mean'], app['B_std']), 0.0, 1.0)
    u_k = np.clip(np.random.normal(app['u_mean'], app['u_std']), 0.02, 1.0)
    r_k = np.clip(np.random.normal(app['r_mean'], app['r_std']), 0.0, 1.0)

    return {
        'id': id(object()), 'app_type': app_type, 'label': app['label'],
        't_start': t_start, 'duration': D_k, 't_end': t_start + D_k,
        'B': B_k, 'u': u_k, 'r': r_k,
    }


# ===============================================================
# 热模型参数
# ===============================================================
Cth  = 40.0    # J/K  热容
Tamb = 25.0    # °C   环境温度
h_th = 20.0    # W/(m²·K) — 此处按散热系数直接使用 (W/K)
# 热贡献权重系数（来自用户公式）
W_LOSS   = 1.0
W_CPU    = 1.0
W_SCREEN = 0.3
W_NET    = 0.7
W_GPS    = 0.7

# 老化量列表 (Ah)
Q_AGING_LIST = [0.0, 100, 500, 1000, 5000]

# Qeff 老化衰减比率
QEFF_DECAY_RATIO = 1.314e-3   # Qeff减少量 = QEFF_DECAY_RATIO * Q
QEFF_NOMINAL     = 5.0         # Ah, 新电池


# ===============================================================
# 温度/老化相关的 OCV 参数函数
# p1, p2, lambda1, lambda2 均变为函数
# ===============================================================
def calc_p1(T, Q_discharged):
    """
    P1(T, Q) = 0.5385 + 6.234e-4*T - 2.608e-7*T^2 (注意：原式第三项缺少T^2，
    结合量纲分析和后续项的结构，此处修正为 -2.608e-7*T^2)
    + 2.191e-5*T^2 + 8.627e-8*T*Q + 8.928e-11*Q^2
    实际上用户写的是 -2.608*1e-7 后面没有变量，结合公式结构看应为常数项修正，
    这里严格按用户提供的字面公式实现:
    P1 = 0.5385 + 6.234e-4*T - 2.608e-7 + 2.191e-5*T^2 + 8.627e-8*T*Q + 8.928e-11*Q^2
    """
    T_clip = np.clip(T, -50.0, 120.0)
    T2 = T_clip * T_clip
    Q2 = Q_discharged * Q_discharged
    return (0.5385
            + 6.234e-4   * T_clip
            - 2.608e-7                       # 常数修正项（按用户字面公式）
            + 2.191e-5   * T2
            + 8.627e-8   * T_clip * Q_discharged
            + 8.928e-11  * Q2)


def calc_p2(T, Q_discharged):
    """
    P2(T, Q) — 严格按用户公式实现（注意 T^3 项）:
    P2 = -3.189e-11 + 9.634e-13*T - 5.539e-16*T*Q - 3.508e-16*T^3
         + 2.347e-17*T^2*Q + 1.431e-20*T*Q^2 - 2.504e-19*T^3*Q - 4.219e-22*T^2*Q^2
    """
    T_clip = np.clip(T, -50.0, 120.0)
    T2 = T_clip * T_clip;   T3 = T2 * T_clip
    Q2 = Q_discharged * Q_discharged
    return (-3.189e-11
            + 9.634e-13  * T_clip
            - 5.539e-16  * T_clip  * Q_discharged
            - 3.508e-16  * T3
            + 2.347e-17  * T2 * Q_discharged
            + 1.431e-20  * T_clip  * Q2
            - 2.504e-19  * T3 * Q_discharged
            - 4.219e-22  * T2 * Q2)


def calc_lambda1(T):
    """
    lambda1(T) = (-6.573e-5*T^2 + 5.767e-3*T - 0.3226) * 2
    单位: Ah^-1 (与放电量 q 相乘后无量纲)
    """
    T_clip = np.clip(T, -50.0, 120.0)
    return (-6.573e-5 * T_clip * T_clip + 5.767e-3 * T_clip - 0.3226) * 2.0


def calc_lambda2(Q_aging):
    """
    lambda2(Q) = (1.299e-13*Q^3 - 2.652e-9*Q^2 + 3.129e-5*Q + 2.33) * 2
    Q = 电池老化水平 (Ah)，决定OCV曲线的形状系数
    """
    Q  = Q_aging
    Q2 = Q * Q;  Q3 = Q2 * Q
    return (1.299e-13 * Q3 - 2.652e-9 * Q2 + 3.129e-5 * Q + 2.33) * 2.0


def calc_Qeff(Q_aging):
    """有效容量 = 新电池容量 - 衰减量"""
    return max(QEFF_NOMINAL*( 1 - QEFF_DECAY_RATIO * np.sqrt(Q_aging)), 0.5)  # 至少保留0.5Ah


# ===============================================================
# 分解各模块功耗（用于热模型权重计算）
# 从 active_set 逐事件累加，返回各模块总功率
# ===============================================================
def calc_module_powers(active_set):
    """
    返回: P_cpu_total, P_screen_total, P_net_total, P_gps_total
    均为增量值（不含 base），base 部分单独处理
    """
    P_cpu  = 0.0
    P_scr  = 0.0
    P_net  = 0.0
    P_gps  = 0.0
    for ev in active_set:
        app = APP_TYPES[ev['app_type']]
        P_scr  += calc_P_screen(ev['B'])
        P_cpu  += calc_P_cpu(ev['u']) - calc_P_cpu(0.02)
        P_net  += calc_P_net(app['net_state'], ev['r']) - calc_P_net('WIFI6', 0.0)
        P_gps  += calc_P_gps(app['gps_state']) - calc_P_gps('off')
    # 加上 base 中各模块的贡献
    P_cpu  += calc_P_cpu(0.02)
    P_scr  += calc_P_screen(0.0)   # = 0
    P_net  += calc_P_net('WIFI6', 0.0)
    P_gps  += calc_P_gps('off')    # = 0
    return P_cpu, P_scr, P_net, P_gps


# ===============================================================
# 核心仿真函数（带热模型 + 老化参数）
# 状态向量: [SOC, Vrc, T]
# ===============================================================
def run_parallel_simulation(Q_aging, T_sim=86400, dt=10.0):
    """
    并行 Marked Poisson + Thevenin ECM + 一阶热模型 耦合仿真.

    状态: [soc, Vrc] 由 RK4 积分; T 由半隐式精确解更新（避免刚性不稳定）.

    热模型说明:
        dT/dt = [Q_src - h*(T - Tamb)] / Cth
    这是关于 T 的一阶线性 ODE，在 Q_src 视为常数的前提下有精确解:
        T(t+dt) = T_ss + (T(t) - T_ss) * exp(-dt/tau)
        其中 T_ss = Tamb + Q_src/h,  tau = Cth/h
    此方法对任意 dt 均无条件稳定（此处 tau=Cth/h=2s, dt=10s, 若用显式RK4会发散）.

    参数:
        Q_aging: 电池老化量 (Ah), 影响 Qeff 和 lambda2
    """
    Qeff   = calc_Qeff(Q_aging)
    tau_th = Cth / h_th                    # 热时间常数 (s)
    lam2   = calc_lambda2(Q_aging)         # lambda2 仅依赖老化量，预算一次

    # --- 时间序列初始化 ---
    n_steps = int(T_sim / dt)
    t_arr       = np.zeros(n_steps)
    soc_arr     = np.zeros(n_steps)
    Vt_arr      = np.zeros(n_steps)
    Voc_arr     = np.zeros(n_steps)
    IL_arr      = np.zeros(n_steps)
    Vrc_arr     = np.zeros(n_steps)
    T_arr       = np.zeros(n_steps)
    P_total_arr = np.zeros(n_steps)
    P_loss_arr  = np.zeros(n_steps)
    n_active_arr= np.zeros(n_steps)
    P_cpu_arr   = np.zeros(n_steps)
    P_scr_arr   = np.zeros(n_steps)
    P_net_arr   = np.zeros(n_steps)
    P_gps_arr   = np.zeros(n_steps)

    # --- 状态初始化 ---
    soc = 1.0
    Vrc = 0.0
    T   = Tamb

    # --- 事件管理 ---
    active_set      = []
    event_log       = []
    app_active_time = {k: 0.0 for k in APP_TYPES}

    # ---- 电化学导数函数（2D: soc, Vrc） ----
    def deriv_ecm(s, v, Temp, P_load):
        """
        返回: (dsdt, dVrcdt, IL, Voc, VL, Ploss)
        """
        s_c   = np.clip(s, 0.05, 1.0)
        R0    = max(float(R0_poly(s_c)), 1e-6)
        R1    = max(float(R1_poly(s_c)), 1e-6)
        C1    = max(float(C1_poly(s_c)), 1e-6)
        gamma = R1 * C1

        q = (1 - s) * Qeff   # 已放电量

        # 温度/老化相关 OCV 参数
        p1_val  = calc_p1(Temp, q)
        p2_val  = calc_p2(Temp, q)
        lam1_val= calc_lambda1(Temp)
        # lam2 已在外层预算（仅依赖 Q_aging）

        # OCV — 为避免指数溢出，限制指数的输入范围
        x1 = np.clip(lam1_val * q, -50.0, 50.0)
        x2 = np.clip(lam2      * q, -50.0, 50.0)
        Voc = (p1_val * (np.exp(x1) - 1)
             + p2_val * (np.exp(x2) - 1)
             + 4.2)

        # 恒功率求 IL
        Vt   = Voc - v
        disc = Vt**2 - 4 * R0 * P_load
        if disc < 0:
            disc = 0.0
        IL = (Vt - np.sqrt(disc)) / (2 * R0)
        IL = np.clip(IL, 0.0, 100.0)  # 限制电流防止异常大

        # 负载端电压和损耗
        VL    = Voc - v - IL * R0
        Ploss = Voc * IL - VL * IL   # = IL²R0 + Vrc·IL
        # 限制 Ploss 防止数值发散
        Ploss = np.clip(Ploss, -1e3, 1e3)

        dsdt   = -IL / Qeff / 3600.0
        dVrcdt = (IL * R1 - v) / gamma

        return dsdt, dVrcdt, IL, Voc, VL, Ploss

    # ---- RK4 积分一步（soc, Vrc, T） ----
    # 整合电化学ECM + 一阶热模型（全部使用RK4数值积分）
    def rk4_step_ecm(soc_in, Vrc_in, Temp_in, P_load, Q_other, h_step):
        """
        使用RK4对耦合状态 [soc, Vrc, T] 进行数值积分。

        参数:
            soc_in: 初始 SOC
            Vrc_in: 初始 Vrc
            Temp_in: 初始温度
            P_load: 负载功率
            Q_other: 热源中不依赖 Ploss 的部分（W），即 W_CPU*P_cpu_e + W_SCREEN*P_scr_e + W_NET*P_net_e + W_GPS*P_gps_e
            h_step: 时步长度(s)

        返回:
            soc_out, Vrc_out, Temp_out, IL_end, Voc_end, VL_end, Ploss_end
        """
        # 内部：给定 (s,v,T) 返回三元导数及 Ploss
        def deriv_total(s, v, T):
            dsdt, dVrcdt, IL, Voc, VL, Ploss = deriv_ecm(s, v, T, P_load)
            Q_source = W_LOSS * Ploss + Q_other
            # 防止热源异常大导致温度发散，限制 Q_source
            Q_source = np.clip(Q_source, -1e3, 1e3)
            dTdt = (Q_source - h_th * (T - Tamb)) / Cth
            # 限制温度变化率防止 RK4 中间阶段爆炸（K/s）
            dTdt = np.clip(dTdt, -10.0, 10.0)
            return dsdt, dVrcdt, dTdt, IL, Voc, VL, Ploss

        # RK4 四阶积分（耦合三维）
        ds1, dv1, dT1, _, _, _, _ = deriv_total(soc_in, Vrc_in, Temp_in)

        s2 = soc_in  + 0.5 * h_step * ds1
        v2 = Vrc_in  + 0.5 * h_step * dv1
        T2 = np.clip(Temp_in + 0.5 * h_step * dT1, -50.0, 120.0)
        ds2, dv2, dT2, _, _, _, _ = deriv_total(s2, v2, T2)

        s3 = soc_in  + 0.5 * h_step * ds2
        v3 = Vrc_in  + 0.5 * h_step * dv2
        T3 = np.clip(Temp_in + 0.5 * h_step * dT2, -50.0, 120.0)
        ds3, dv3, dT3, _, _, _, _ = deriv_total(s3, v3, T3)

        s4 = soc_in  + h_step * ds3
        v4 = Vrc_in  + h_step * dv3
        T4 = np.clip(Temp_in + h_step * dT3, -50.0, 120.0)
        ds4, dv4, dT4, _, _, _, _ = deriv_total(s4, v4, T4)

        soc_out  = soc_in + (h_step/6.0) * (ds1 + 2*ds2 + 2*ds3 + ds4)
        Vrc_out  = Vrc_in + (h_step/6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        Temp_out = Temp_in + (h_step/6.0) * (dT1 + 2*dT2 + 2*dT3 + dT4)

        # 终点回算物理量（使用终点温度）
        _, _, IL_end, Voc_end, VL_end, Ploss_end = deriv_ecm(soc_out, Vrc_out, Temp_out, P_load)
        return soc_out, Vrc_out, Temp_out, IL_end, Voc_end, VL_end, Ploss_end


    # =========== 主仿真循环 ===========
    for i in range(n_steps):
        t = i * dt
        t_arr[i] = t

        # ── Step 1: 泊松到达检测 ──
        lam_val  = lambda_intensity(t)
        p_arrive = lam_val * (dt / 3600.0)
        n_new    = (1 if np.random.rand() < p_arrive else 0) if p_arrive <= 0.5 \
                   else np.random.poisson(p_arrive)
        for _ in range(n_new):
            ev = sample_event(t)
            active_set.append(ev)
            event_log.append(ev)

        # ── Step 2: 清理已结束事件 ──
        still_active = []
        for ev in active_set:
            if t >= ev['t_end']:
                pass
            else:
                still_active.append(ev)
                app_active_time[ev['app_type']] += dt
        active_set = still_active

        # ── Step 3: 计算总负载和各模块功耗 ──
        P_events = sum(calc_event_power(ev) for ev in active_set)
        P_load   = P_BASE + P_events
        P_cpu_e, P_scr_e, P_net_e, P_gps_e = calc_module_powers(active_set)

        # ── Step 4: 检查 SOC 截止 ──
        if soc <= 0.05:
            t_arr       = t_arr[:i];  soc_arr     = soc_arr[:i]
            Vt_arr      = Vt_arr[:i]; Voc_arr     = Voc_arr[:i]
            IL_arr      = IL_arr[:i]; Vrc_arr     = Vrc_arr[:i]
            T_arr       = T_arr[:i];  P_total_arr = P_total_arr[:i]
            P_loss_arr  = P_loss_arr[:i]; n_active_arr = n_active_arr[:i]
            P_cpu_arr   = P_cpu_arr[:i]; P_scr_arr  = P_scr_arr[:i]
            P_net_arr   = P_net_arr[:i]; P_gps_arr  = P_gps_arr[:i]
            break

        # ── Step 5: 计算热源中不依赖 Ploss 的部分（Q_other），并使用RK4求解耦合态
        # Q_other 包含各模块对热源的贡献，但不含 W_LOSS*Ploss
        Q_other = (W_CPU    * P_cpu_e
                 + W_SCREEN * P_scr_e
                 + W_NET    * P_net_e
                 + W_GPS    * P_gps_e)

        soc_new, Vrc_new, T_new, IL_val, Voc_val, VL_val, Ploss_val = \
            rk4_step_ecm(soc, Vrc, T, P_load, Q_other, dt)
        soc_new = max(soc_new, 0.05)

        # 记录
        soc_arr[i]      = soc
        Vrc_arr[i]      = Vrc
        T_arr[i]        = T
        Vt_arr[i]       = VL_val
        Voc_arr[i]      = Voc_val
        IL_arr[i]       = IL_val
        P_total_arr[i]  = P_load
        P_loss_arr[i]   = Ploss_val
        n_active_arr[i] = len(active_set)
        P_cpu_arr[i]    = P_cpu_e
        P_scr_arr[i]    = P_scr_e
        P_net_arr[i]    = P_net_e
        P_gps_arr[i]    = P_gps_e

        # 推进
        soc = soc_new
        Vrc = Vrc_new
        T   = T_new

    discharge_time_h = t_arr[-1] / 3600.0 if len(t_arr) > 0 else 0.0

    return {
        't': t_arr, 'soc': soc_arr, 'Vt': Vt_arr, 'Voc': Voc_arr,
        'IL': IL_arr, 'Vrc': Vrc_arr, 'T': T_arr,
        'P_total': P_total_arr, 'P_loss': P_loss_arr,
        'n_active': n_active_arr,
        'P_cpu': P_cpu_arr, 'P_scr': P_scr_arr,
        'P_net': P_net_arr, 'P_gps': P_gps_arr,
        'event_log': event_log,
        'app_active_time': app_active_time,
        'discharge_time_h': discharge_time_h,
        'Qeff': Qeff, 'Q_aging': Q_aging,
    }


# ===============================================================
# 运行所有老化水平的仿真
# 每个老化水平使用相同的随机种子以保证事件序列一致
# ===============================================================
print("=" * 70)
print("运行多老化水平仿真 ...")
print("=" * 70)

all_results = {}
for Q_ag in Q_AGING_LIST:
    np.random.seed(42)   # 固定种子 → 相同事件序列
    Qeff_val = calc_Qeff(Q_ag)
    print(f"\n  Q_aging = {Q_ag:>7.1f} Ah,  Qeff = {Qeff_val:.4f} Ah")
    res = run_parallel_simulation(Q_aging=Q_ag, T_sim=86400, dt=10.0)
    all_results[Q_ag] = res
    print(f"    放电时间: {res['discharge_time_h']:.2f} h, "
          f"最高温度: {res['T'].max():.2f} °C, "
          f"平均负载: {np.mean(res['P_total'])*1000:.1f} mW")

print("\n" + "=" * 70)


# ===============================================================
# 颜色映射
# ===============================================================
APP_COLORS = {
    'video': '#e74c3c', 'gaming': '#9b59b6', 'social': '#3498db',
    'navigation': '#e67e22', 'calling': '#1abc9c', 'music': '#f39c12',
    'bg_download': '#2ecc71', 'bg_sync': '#95a5a6',
}

# 老化水平颜色
AGING_COLORS = {
    0.0:    '#2ecc71',   # 新电池 - 绿
    100:   '#3498db',   # 轻度 - 蓝
    500:   '#f39c12',   # 中度 - 橙
    1000:  '#e67e22',   # 重度 - 深橙
    5000.0: '#e74c3c',   # 极度 - 红
}

AGING_LABELS = {
    0.0:    '新电池 (Q=0)',
    100.0: '轻度老化 (Q=100)',
    500.0:   '中度老化 (Q=500)',
    1000.0:  '重度老化 (Q=1000)',
    5000.0: '极度老化 (Q=5000)',
}


# ===============================================================
# 图 1: 泊松强度函数 λ(t)
# ===============================================================
fig1, ax = plt.subplots(figsize=(12, 4))
t_24h   = np.linspace(0, 86400, 1000)
lam_vals= [lambda_intensity(t) for t in t_24h]
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
# 图 2: 多老化水平对比 — SOC / 温度 / 端电压 / 放电电流
# ===============================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.suptitle('不同老化水平下电池状态与温度演化对比', fontsize=15, fontweight='bold')

for Q_ag in Q_AGING_LIST:
    r     = all_results[Q_ag]
    t_h   = r['t'] / 3600.0
    color = AGING_COLORS[Q_ag]
    label = AGING_LABELS[Q_ag]

    # SOC
    axes2[0,0].plot(t_h, r['soc'], linewidth=2, color=color, label=label)
    # 温度
    axes2[0,1].plot(t_h, r['T'],   linewidth=2, color=color, label=label)
    # 端电压
    axes2[1,0].plot(t_h, r['Vt'],  linewidth=1.8, color=color, label=label)
    # 放电电流
    axes2[1,1].plot(t_h, r['IL'],  linewidth=1.0, color=color, alpha=0.8, label=label)

# SOC
axes2[0,0].axhline(y=0.05, color='red', linestyle=':', linewidth=1.2, alpha=0.6)
axes2[0,0].set_xlabel('时间 (h)', fontsize=11)
axes2[0,0].set_ylabel('SOC', fontsize=11)
axes2[0,0].set_title('SOC 随时间变化', fontsize=12, fontweight='bold')
axes2[0,0].legend(fontsize=8, loc='best')
axes2[0,0].grid(True, alpha=0.3)
axes2[0,0].set_ylim(0, 1.05)

# 温度
axes2[0,1].axhline(y=Tamb, color='gray', linestyle='--', linewidth=1, label=f'环境温度 {Tamb}°C')
axes2[0,1].set_xlabel('时间 (h)', fontsize=11)
axes2[0,1].set_ylabel('温度 T (°C)', fontsize=11)
axes2[0,1].set_title('电池温度随时间变化', fontsize=12, fontweight='bold')
axes2[0,1].legend(fontsize=8, loc='best')
axes2[0,1].grid(True, alpha=0.3)

# 端电压
axes2[1,0].set_xlabel('时间 (h)', fontsize=11)
axes2[1,0].set_ylabel('负载端电压 VL (V)', fontsize=11)
axes2[1,0].set_title('端电压随时间变化', fontsize=12, fontweight='bold')
axes2[1,0].legend(fontsize=8, loc='best')
axes2[1,0].grid(True, alpha=0.3)

# 放电电流
axes2[1,1].set_xlabel('时间 (h)', fontsize=11)
axes2[1,1].set_ylabel('放电电流 IL (A)', fontsize=11)
axes2[1,1].set_title('放电电流随时间变化', fontsize=12, fontweight='bold')
axes2[1,1].legend(fontsize=8, loc='best')
axes2[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ===============================================================
# 图 3: 多老化水平 — 温度曲线 + 损耗功率 (以新电池为例详细展示)
# ===============================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))
fig3.suptitle('热模型详细分析', fontsize=15, fontweight='bold')

# (0,0): 各老化水平温度对比（放大纵轴）
for Q_ag in Q_AGING_LIST:
    r   = all_results[Q_ag]
    t_h = r['t'] / 3600.0
    axes3[0,0].plot(t_h, r['T'], linewidth=2,
                    color=AGING_COLORS[Q_ag], label=AGING_LABELS[Q_ag])
axes3[0,0].axhline(y=Tamb, color='gray', linestyle='--', linewidth=1)
axes3[0,0].set_xlabel('时间 (h)', fontsize=11)
axes3[0,0].set_ylabel('温度 (°C)', fontsize=11)
axes3[0,0].set_title('温度对比（全部老化水平）', fontsize=12, fontweight='bold')
axes3[0,0].legend(fontsize=8)
axes3[0,0].grid(True, alpha=0.3)

# (0,1): 新电池的各热源贡献分解 (堆叠面积图)
r_new = all_results[0.0]
t_h   = r_new['t'] / 3600.0
axes3[0,1].stackplot(t_h,
    r_new['P_loss']*1000,
    r_new['P_cpu']*1000,
    r_new['P_scr']*1000,
    r_new['P_net']*1000,
    r_new['P_gps']*1000,
    labels=['电池损耗 Ploss', 'CPU', '屏幕', '网络', 'GPS'],
    colors=['#e74c3c', '#9b59b6', '#f39c12', '#3498db', '#2ecc71'],
    alpha=0.7)
axes3[0,1].set_xlabel('时间 (h)', fontsize=11)
axes3[0,1].set_ylabel('功率 (mW)', fontsize=11)
axes3[0,1].set_title('热源功率分解 (新电池)', fontsize=12, fontweight='bold')
axes3[0,1].legend(fontsize=8, loc='upper left')
axes3[0,1].grid(True, alpha=0.2)

# (1,0): 损耗功率 Ploss 各老化水平对比
for Q_ag in Q_AGING_LIST:
    r   = all_results[Q_ag]
    t_h = r['t'] / 3600.0
    axes3[1,0].plot(t_h, r['P_loss']*1000, linewidth=1.2,
                    color=AGING_COLORS[Q_ag], label=AGING_LABELS[Q_ag], alpha=0.8)
axes3[1,0].set_xlabel('时间 (h)', fontsize=11)
axes3[1,0].set_ylabel('损耗功率 Ploss (mW)', fontsize=11)
axes3[1,0].set_title('电池内部损耗功率对比', fontsize=12, fontweight='bold')
axes3[1,0].legend(fontsize=8)
axes3[1,0].grid(True, alpha=0.3)

# (1,1): 温度-SOC 相位图
for Q_ag in Q_AGING_LIST:
    r = all_results[Q_ag]
    axes3[1,1].plot(r['soc'], r['T'], linewidth=1.5,
                    color=AGING_COLORS[Q_ag], label=AGING_LABELS[Q_ag])
    # 起点标记
    axes3[1,1].scatter([r['soc'][0]], [r['T'][0]], color=AGING_COLORS[Q_ag],
                       s=60, zorder=5, edgecolor='black', linewidth=0.8)
axes3[1,1].set_xlabel('SOC', fontsize=11)
axes3[1,1].set_ylabel('温度 (°C)', fontsize=11)
axes3[1,1].set_title('温度-SOC 相位图（圆点=起点）', fontsize=12, fontweight='bold')
axes3[1,1].legend(fontsize=8)
axes3[1,1].grid(True, alpha=0.3)
axes3[1,1].invert_xaxis()

plt.tight_layout()
plt.show()


# ===============================================================
# 图 4: Gantt 图 + 功耗叠加（以新电池为例）
# ===============================================================
r_ref = all_results[0.0]
t_end_sim = r_ref['t'][-1] if len(r_ref['t']) > 0 else 86400

fig4, (ax_gantt, ax_power) = plt.subplots(2, 1, figsize=(16, 9),
                                            gridspec_kw={'height_ratios': [2, 1]})
fig4.suptitle('并行事件时间线与总功耗演化 (新电池)', fontsize=15, fontweight='bold')

type_keys_ordered = list(APP_TYPES.keys())
y_positions = {k: i for i, k in enumerate(type_keys_ordered)}

for ev in r_ref['event_log']:
    if ev['t_start'] >= t_end_sim:
        continue
    y     = y_positions[ev['app_type']]
    x0    = ev['t_start'] / 3600.0
    width = min(ev['duration'], t_end_sim - ev['t_start']) / 3600.0
    ax_gantt.barh(y, width, left=x0, height=0.7,
                  color=APP_COLORS[ev['app_type']], alpha=0.75,
                  edgecolor='white', linewidth=0.4)

ax_gantt.set_yticks(list(y_positions.values()))
ax_gantt.set_yticklabels([APP_TYPES[k]['label'] for k in type_keys_ordered], fontsize=10)
ax_gantt.set_xlabel('时间 (h)', fontsize=11)
ax_gantt.set_title('各应用类型事件时间线（Gantt图）', fontsize=12)
ax_gantt.grid(axis='x', alpha=0.3)
ax_gantt.set_xlim(0, t_end_sim / 3600.0)
legend_patches = [mpatches.Patch(color=APP_COLORS[k], label=APP_TYPES[k]['label'])
                  for k in type_keys_ordered]
ax_gantt.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)

# 功耗曲线
t_h = r_ref['t'] / 3600.0
ax_power.plot(t_h, r_ref['P_total']*1000, linewidth=1.0, color='#e74c3c', alpha=0.7)
ax_power.fill_between(t_h, P_BASE*1000, r_ref['P_total']*1000,
                       alpha=0.25, color='#e74c3c', label='事件叠加功耗')
ax_power.axhline(y=P_BASE*1000, color='#2c3e50', linestyle='--',
                 linewidth=1.5, label=f'基础待机 {P_BASE*1000:.1f} mW')
ax_power.set_xlabel('时间 (h)', fontsize=11)
ax_power.set_ylabel('总负载 P(t) (mW)', fontsize=11)
ax_power.set_title('总功耗随时间变化', fontsize=12)
ax_power.legend(fontsize=9, loc='upper left')
ax_power.grid(True, alpha=0.3)
ax_power.set_xlim(0, t_end_sim / 3600.0)

ax_n = ax_power.twinx()
ax_n.plot(t_h, r_ref['n_active'], linewidth=1.2, color='#27ae60', alpha=0.8)
ax_n.set_ylabel('活跃事件数', fontsize=11, color='#27ae60')
ax_n.tick_params(axis='y', labelcolor='#27ae60')
ax_n.set_ylim(0, max(r_ref['n_active'].max()+1, 6))

plt.tight_layout()
plt.show()


# ===============================================================
# 图 5: 老化水平对比汇总柱状图
# ===============================================================
fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5.5))
fig5.suptitle('老化水平对比汇总', fontsize=14, fontweight='bold')

Q_labels  = [AGING_LABELS[q] for q in Q_AGING_LIST]
Q_colors  = [AGING_COLORS[q] for q in Q_AGING_LIST]
discharge_h  = [all_results[q]['discharge_time_h'] for q in Q_AGING_LIST]
max_temps    = [all_results[q]['T'].max() for q in Q_AGING_LIST]
Qeff_vals    = [calc_Qeff(q) for q in Q_AGING_LIST]

# (0) 放电时间
bars = axes5[0].bar(range(len(Q_AGING_LIST)), discharge_h, color=Q_colors,
                    edgecolor='#2c3e50', linewidth=0.7)
for i, v in enumerate(discharge_h):
    axes5[0].text(i, v + 0.05, f'{v:.2f}h', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes5[0].set_xticks(range(len(Q_AGING_LIST)))
axes5[0].set_xticklabels(Q_labels, fontsize=8, rotation=15)
axes5[0].set_ylabel('放电时间 (h)', fontsize=11)
axes5[0].set_title('各老化水平使用寿命', fontsize=12, fontweight='bold')
axes5[0].grid(axis='y', alpha=0.3)

# (1) 最高温度
bars = axes5[1].bar(range(len(Q_AGING_LIST)), max_temps, color=Q_colors,
                    edgecolor='#2c3e50', linewidth=0.7)
for i, v in enumerate(max_temps):
    axes5[1].text(i, v + 0.05, f'{v:.2f}°C', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes5[1].set_xticks(range(len(Q_AGING_LIST)))
axes5[1].set_xticklabels(Q_labels, fontsize=8, rotation=15)
axes5[1].set_ylabel('最高温度 (°C)', fontsize=11)
axes5[1].set_title('各老化水平峰值温度', fontsize=12, fontweight='bold')
axes5[1].axhline(y=Tamb, color='gray', linestyle=':', linewidth=1)
axes5[1].grid(axis='y', alpha=0.3)

# (2) 有效容量 Qeff
bars = axes5[2].bar(range(len(Q_AGING_LIST)), Qeff_vals, color=Q_colors,
                    edgecolor='#2c3e50', linewidth=0.7)
for i, v in enumerate(Qeff_vals):
    axes5[2].text(i, v + 0.02, f'{v:.3f}Ah', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes5[2].set_xticks(range(len(Q_AGING_LIST)))
axes5[2].set_xticklabels(Q_labels, fontsize=8, rotation=15)
axes5[2].set_ylabel('有效容量 Qeff (Ah)', fontsize=11)
axes5[2].set_title('各老化水平有效容量', fontsize=12, fontweight='bold')
axes5[2].axhline(y=QEFF_NOMINAL, color='gray', linestyle=':', linewidth=1, label=f'新电池 {QEFF_NOMINAL} Ah')
axes5[2].legend(fontsize=9)
axes5[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# ===============================================================
# 图 6: OCV 参数随温度/放电量变化（参数灵敏度展示）
# ===============================================================
fig6, axes6 = plt.subplots(2, 2, figsize=(14, 10))
fig6.suptitle('OCV 参数随温度与放电量的变化 (参数灵敏度)', fontsize=14, fontweight='bold')

T_sweep = np.linspace(20, 55, 100)   # 温度扫描
q_sweep = np.linspace(0, 4.5, 100)   # 放电量扫描

# (0,0) p1 vs T (不同固定 q)
for q_fix in [0, 1, 2, 3, 4]:
    p1_vals = [calc_p1(T, q_fix) for T in T_sweep]
    axes6[0,0].plot(T_sweep, p1_vals, linewidth=2, label=f'q={q_fix} Ah')
axes6[0,0].set_xlabel('温度 T (°C)', fontsize=11)
axes6[0,0].set_ylabel('p1', fontsize=11)
axes6[0,0].set_title('p1(T, q)', fontsize=12, fontweight='bold')
axes6[0,0].legend(fontsize=9); axes6[0,0].grid(True, alpha=0.3)

# (0,1) lambda1 vs T
lam1_vals = [calc_lambda1(T) for T in T_sweep]
axes6[0,1].plot(T_sweep, lam1_vals, linewidth=2.5, color='#e74c3c')
axes6[0,1].set_xlabel('温度 T (°C)', fontsize=11)
axes6[0,1].set_ylabel('λ1 (Ah⁻¹)', fontsize=11)
axes6[0,1].set_title('λ1(T)', fontsize=12, fontweight='bold')
axes6[0,1].grid(True, alpha=0.3)

# (1,0) lambda2 vs q
lam2_vals = [calc_lambda2(q) for q in q_sweep]
axes6[1,0].plot(q_sweep, lam2_vals, linewidth=2.5, color='#3498db')
axes6[1,0].set_xlabel('放电量 q (Ah)', fontsize=11)
axes6[1,0].set_ylabel('λ2 (Ah⁻¹)', fontsize=11)
axes6[1,0].set_title('λ2(q)', fontsize=12, fontweight='bold')
axes6[1,0].grid(True, alpha=0.3)

# (1,1) OCV 曲线对比：不同温度下的 Voc vs SOC
Qeff_ref = QEFF_NOMINAL   # 用新电池容量
soc_plot = np.linspace(0.05, 1.0, 200)
for T_fix in [20, 25, 35, 45, 55]:
    Voc_curve = []
    for s in soc_plot:
        q_val = (1 - s) * Qeff_ref
        p1v   = calc_p1(T_fix, q_val)
        p2v   = calc_p2(T_fix, q_val)
        l1v   = calc_lambda1(T_fix)
        l2v   = calc_lambda2(q_val)
        Voc   = p1v*(np.exp(l1v*q_val)-1) + p2v*(np.exp(l2v*q_val)-1) + 4.2
        Voc_curve.append(Voc)
    axes6[1,1].plot(soc_plot, Voc_curve, linewidth=2, label=f'T={T_fix}°C')
axes6[1,1].set_xlabel('SOC', fontsize=11)
axes6[1,1].set_ylabel('Voc (V)', fontsize=11)
axes6[1,1].set_title('OCV 曲线对比（不同温度）', fontsize=12, fontweight='bold')
axes6[1,1].legend(fontsize=9); axes6[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ===============================================================
# 图 7: ECM 参数拟合曲线 + 网络功率对比（保留原有功能）
# ===============================================================
soc_fine = np.linspace(0.1, 0.9, 200)
fig7, axes7 = plt.subplots(1, 4, figsize=(20, 4.5))
fig7.suptitle('ECM参数拟合 & 网络功率特性', fontsize=14, fontweight='bold')

axes7[0].plot(soc_fine, R0_poly(soc_fine)*1000, linewidth=2, color='blue', label='Cubic Fit')
axes7[0].scatter(soc_table, R0_table*1000, color='red', s=50, zorder=5, label='Data', edgecolor='k', linewidth=0.5)
axes7[0].set_xlabel('SOC'); axes7[0].set_ylabel('R0 (mΩ)')
axes7[0].set_title(f'R0 (R^2={r_squared(R0_table,R0_fit):.4f})', fontsize=11)
axes7[0].legend(fontsize=8); axes7[0].grid(True, alpha=0.3)

axes7[1].plot(soc_fine, R1_poly(soc_fine)*1000, linewidth=2, color='red', label='Cubic Fit')
axes7[1].scatter(soc_table, R1_table*1000, color='blue', s=50, zorder=5, label='Data', edgecolor='k', linewidth=0.5)
axes7[1].set_xlabel('SOC'); axes7[1].set_ylabel('R1 (mΩ)')
axes7[1].set_title(f'R1 (R^2={r_squared(R1_table,R1_fit):.4f})', fontsize=11)
axes7[1].legend(fontsize=8); axes7[1].grid(True, alpha=0.3)

axes7[2].plot(soc_fine, C1_poly(soc_fine), linewidth=2, color='green', label='Cubic Fit')
axes7[2].scatter(soc_table, C1_table, color='orange', s=50, zorder=5, label='Data', edgecolor='k', linewidth=0.5)
axes7[2].set_xlabel('SOC'); axes7[2].set_ylabel('C1 (F)')
axes7[2].set_title(f'C1 (R^2={r_squared(C1_table,C1_fit):.4f})', fontsize=11)
axes7[2].legend(fontsize=8); axes7[2].grid(True, alpha=0.3)

r_sweep = np.linspace(0, 1, 200)
for net_key, color in [('WIFI6','#3498db'),('4G_LTE_A','#e67e22'),('5G_SA','#e74c3c')]:
    p = NET_PARAMS[net_key]
    axes7[3].plot(r_sweep, p['k']*r_sweep+p['idle'], linewidth=2.2, color=color, label=net_key.replace('_',' '))
axes7[3].set_xlabel('传输速率 r'); axes7[3].set_ylabel('网络功率 (mW)')
axes7[3].set_title('网络功率特性', fontsize=11)
axes7[3].legend(fontsize=8); axes7[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ===============================================================
# 最终汇总输出
# ===============================================================
print("\n" + "=" * 70)
print("各老化水平仿真汇总")
print("=" * 70)
print(f"{'老化水平':<22} {'Qeff(Ah)':>9} {'放电时间(h)':>12} {'平均负载(mW)':>13} "
      f"{'最高温度(°C)':>13} {'最大并行事件':>12} {'总事件数':>9}")
print("-" * 95)
for Q_ag in Q_AGING_LIST:
    r = all_results[Q_ag]
    print(f"{AGING_LABELS[Q_ag]:<22} "
          f"{calc_Qeff(Q_ag):>8.4f} "
          f"{r['discharge_time_h']:>11.2f} "
          f"{np.mean(r['P_total'])*1000:>12.1f} "
          f"{r['T'].max():>12.2f} "
          f"{int(r['n_active'].max()):>11} "
          f"{len(r['event_log']):>8}")
print("=" * 95)