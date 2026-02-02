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
# 应用类型定义（联合泊松过程 - 每个应用独立泊松过程）
# ===============================================================
APP_TYPES = {
    'video': {
        'label': '视频播放',
        'B_mean': 0.72, 'B_std': 0.08,
        'u_mean': 0.22, 'u_std': 0.04,
        'r_mean': 0.70, 'r_std': 0.10,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'dur_mean': 4800.0, 'dur_std': 2400.0,
        'lambda_weight': 0.22,
    },
    'gaming': {
        'label': '游戏',
        'B_mean': 0.85, 'B_std': 0.05,
        'u_mean': 0.65, 'u_std': 0.08,
        'r_mean': 0.45, 'r_std': 0.12,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'dur_mean': 3600.0, 'dur_std': 1800.0,
        'lambda_weight': 0.15,
    },
    'social': {
        'label': '社交媒体',
        'B_mean': 0.60, 'B_std': 0.10,
        'u_mean': 0.18, 'u_std': 0.05,
        'r_mean': 0.35, 'r_std': 0.08,
        'net_state': '4G_LTE_A', 'gps_state': 'off',
        'dur_mean': 2400.0, 'dur_std': 1200.0,
        'lambda_weight': 0.28,
    },
    'navigation': {
        'label': '导航',
        'B_mean': 0.75, 'B_std': 0.06,
        'u_mean': 0.35, 'u_std': 0.06,
        'r_mean': 0.25, 'r_std': 0.05,
        'net_state': '4G_LTE_A', 'gps_state': 'locating',
        'dur_mean': 1800.0, 'dur_std': 900.0,
        'lambda_weight': 0.10,
    },
    'calling': {
        'label': '通话',
        'B_mean': 0.15, 'B_std': 0.05,
        'u_mean': 0.12, 'u_std': 0.03,
        'r_mean': 0.08, 'r_std': 0.02,
        'net_state': '4G_LTE_A', 'gps_state': 'off',
        'dur_mean': 600.0, 'dur_std': 300.0,
        'lambda_weight': 0.12,
    },
    'music': {
        'label': '音乐播放',
        'B_mean': 0.20, 'B_std': 0.08,
        'u_mean': 0.08, 'u_std': 0.02,
        'r_mean': 0.15, 'r_std': 0.04,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'dur_mean': 7200.0, 'dur_std': 3600.0,
        'lambda_weight': 0.08,
    },
    'bg_download': {
        'label': '后台下载',
        'B_mean': 0.0, 'B_std': 0.0,
        'u_mean': 0.15, 'u_std': 0.03,
        'r_mean': 0.85, 'r_std': 0.08,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'dur_mean': 1200.0, 'dur_std': 600.0,
        'lambda_weight': 0.03,
    },
    'bg_sync': {
        'label': '后台同步',
        'B_mean': 0.0, 'B_std': 0.0,
        'u_mean': 0.05, 'u_std': 0.01,
        'r_mean': 0.20, 'r_std': 0.05,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'dur_mean': 180.0, 'dur_std': 90.0,
        'lambda_weight': 0.02,
    },
}

APP_TYPE_KEYS = list(APP_TYPES.keys())

# 归一化权重
total_weight = sum(APP_TYPES[k]['lambda_weight'] for k in APP_TYPE_KEYS)
for k in APP_TYPE_KEYS:
    APP_TYPES[k]['lambda_weight'] /= total_weight


# ===============================================================
# 联合泊松过程强度函数 λᵢ(t) - 每个应用类型独立
# ===============================================================
def lambda_intensity_per_app(t_sec, app_type):
    """每个应用类型的独立到达率函数 λᵢ(t)"""
    app = APP_TYPES[app_type]
    t_h = (t_sec % 86400) / 3600.0
    
    # 基础昼夜节律模式
    peak1 = 8.0  * np.exp(-0.5 * ((t_h - 13.0) / 3.5)**2)
    peak2 = 10.0 * np.exp(-0.5 * ((t_h - 21.0) / 2.5)**2)
    base_intensity = 0.5 + peak1 + peak2
    
    # 应用特定的时间偏好调制
    if app_type == 'video':
        time_modulation = 1.0 + 0.5 * np.exp(-0.5 * ((t_h - 20.0) / 3.0)**2)
    elif app_type == 'gaming':
        time_modulation = 1.0 + 0.3 * (np.exp(-0.5 * ((t_h - 15.0) / 2.5)**2) + 
                                        np.exp(-0.5 * ((t_h - 22.0) / 2.0)**2))
    elif app_type == 'social':
        time_modulation = 1.0 + 0.2 * (np.exp(-0.5 * ((t_h - 8.0) / 2.0)**2) + 
                                        np.exp(-0.5 * ((t_h - 19.0) / 2.5)**2))
    elif app_type == 'navigation':
        time_modulation = 1.0 + 0.8 * (np.exp(-0.5 * ((t_h - 8.0) / 1.5)**2) + 
                                        np.exp(-0.5 * ((t_h - 18.0) / 1.5)**2))
    elif app_type == 'calling':
        morning_factor = 1.0 / (1.0 + np.exp(-3.0 * (t_h - 9.0)))
        evening_factor = 1.0 / (1.0 + np.exp(3.0 * (t_h - 18.0)))
        time_modulation = 0.8 + 0.5 * morning_factor * evening_factor
    elif app_type == 'music':
        time_modulation = 1.0 + 0.4 * (np.exp(-0.5 * ((t_h - 8.5) / 2.0)**2) + 
                                        np.exp(-0.5 * ((t_h - 14.0) / 3.0)**2))
    elif app_type in ['bg_download', 'bg_sync']:
        if 1 <= t_h <= 6:
            time_modulation = 1.2
        else:
            time_modulation = 0.9
    else:
        time_modulation = 1.0
    
    return base_intensity * app['lambda_weight'] * time_modulation


def lambda_total(t_sec):
    """总到达率（所有独立泊松过程的和）"""
    return sum(lambda_intensity_per_app(t_sec, app) for app in APP_TYPE_KEYS)


# ===============================================================
# 基础待机功耗
# ===============================================================
def calc_P_base():
    return calc_P_screen(0.0) + calc_P_cpu(0.02) + calc_P_net('WIFI6', 0.0) + calc_P_gps('off')

P_BASE = calc_P_base()
print(f"\n基础待机功耗 P_base = {P_BASE*1000:.2f} mW\n")


# ===============================================================
# 事件功耗贡献
# ===============================================================
def calc_event_power(event):
    app = APP_TYPES[event['app_type']]
    P_scr     = calc_P_screen(event['B'])
    P_cpu_inc = calc_P_cpu(event['u'])   - calc_P_cpu(0.02)
    P_net_inc = calc_P_net(app['net_state'], event['r']) - calc_P_net('WIFI6', 0.0)
    P_gps_inc = calc_P_gps(app['gps_state']) - calc_P_gps('off')
    return max(P_scr + P_cpu_inc + P_net_inc + P_gps_inc, 0.0)


# ===============================================================
# 事件采样
# ===============================================================
def sample_event(t_start, app_type):
    """为指定应用类型采样一个事件"""
    app = APP_TYPES[app_type]

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
Cth  = 40.0
Tamb = 25.0
h_th = 20.0

W_LOSS   = 1.0
W_CPU    = 1.0
W_SCREEN = 0.3
W_NET    = 0.7
W_GPS    = 0.7

Q_AGING_LIST = [0.0, 100, 500, 1000, 5000]

QEFF_DECAY_RATIO = 1.314e-3
QEFF_NOMINAL     = 5.0


# ===============================================================
# OCV 参数函数
# ===============================================================
def calc_p1(T, Q_discharged):
    T_clip = np.clip(T, -50.0, 120.0)
    T2 = T_clip * T_clip
    Q2 = Q_discharged * Q_discharged
    return (0.5385
            + 6.234e-4   * T_clip
            - 2.608e-7
            + 2.191e-5   * T2
            + 8.627e-8   * T_clip * Q_discharged
            + 8.928e-11  * Q2)


def calc_p2(T, Q_discharged):
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
    T_clip = np.clip(T, -50.0, 120.0)
    return (-6.573e-5 * T_clip * T_clip + 5.767e-3 * T_clip - 0.3226) * 2.0


def calc_lambda2(Q_aging):
    Q  = Q_aging
    Q2 = Q * Q;  Q3 = Q2 * Q
    return (1.299e-13 * Q3 - 2.652e-9 * Q2 + 3.129e-5 * Q + 2.33) * 2.0


def calc_Qeff(Q_aging):
    return max(QEFF_NOMINAL*( 1 - QEFF_DECAY_RATIO * np.sqrt(Q_aging)), 0.5)


# ===============================================================
# 分解各模块功耗
# ===============================================================
def calc_module_powers(active_set):
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
    P_cpu  += calc_P_cpu(0.02)
    P_scr  += calc_P_screen(0.0)
    P_net  += calc_P_net('WIFI6', 0.0)
    P_gps  += calc_P_gps('off')
    return P_cpu, P_scr, P_net, P_gps


# ===============================================================
# 核心仿真函数（改进版 - 细化时间步长）
# ===============================================================
def run_joint_poisson_simulation(Q_aging, T_sim=86400, dt_sim=1.0, dt_ode=0.1, 
                                   output_decimation=10):
    """
    联合泊松过程 + Thevenin ECM + 热模型耦合仿真（细化时间步长版本）
    
    核心改进：
    1. dt_sim: 仿真主循环时间步长（事件检测、功耗更新）
    2. dt_ode: ODE求解器内部时间步长（RK4积分）
    3. output_decimation: 输出抽取因子（减少存储）
    
    参数:
        Q_aging: 电池老化量 (Ah)
        T_sim: 仿真总时长（秒）
        dt_sim: 仿真时间步长（秒） - 默认1秒
        dt_ode: ODE求解时间步长（秒） - 默认0.1秒
        output_decimation: 每隔多少步输出一次（默认10，即每10步输出1次）
    """
    Qeff   = calc_Qeff(Q_aging)
    lam2   = calc_lambda2(Q_aging)

    # 仿真步数
    n_steps_sim = int(T_sim / dt_sim)
    
    # 输出数组大小（抽取后）
    n_output = n_steps_sim // output_decimation + 1
    
    t_arr       = np.zeros(n_output)
    soc_arr     = np.zeros(n_output)
    Vt_arr      = np.zeros(n_output)
    Voc_arr     = np.zeros(n_output)
    IL_arr      = np.zeros(n_output)
    Vrc_arr     = np.zeros(n_output)
    T_arr       = np.zeros(n_output)
    P_total_arr = np.zeros(n_output)
    P_loss_arr  = np.zeros(n_output)
    n_active_arr= np.zeros(n_output)
    P_cpu_arr   = np.zeros(n_output)
    P_scr_arr   = np.zeros(n_output)
    P_net_arr   = np.zeros(n_output)
    P_gps_arr   = np.zeros(n_output)
    
    arrival_counts = {k: 0 for k in APP_TYPE_KEYS}

    # 状态初始化
    soc = 1.0
    Vrc = 0.0
    T   = Tamb

    # 事件管理
    active_set      = []
    event_log       = []
    app_active_time = {k: 0.0 for k in APP_TYPES}
    
    # 输出计数器
    output_idx = 0

    # 电化学导数函数
    def deriv_ecm(s, v, Temp, P_load):
        s_c   = np.clip(s, 0.05, 1.0)
        R0    = max(float(R0_poly(s_c)), 1e-6)
        R1    = max(float(R1_poly(s_c)), 1e-6)
        C1    = max(float(C1_poly(s_c)), 1e-6)
        gamma = R1 * C1

        q = (1 - s) * Qeff

        p1_val  = calc_p1(Temp, q)
        p2_val  = calc_p2(Temp, q)
        lam1_val= calc_lambda1(Temp)

        x1 = np.clip(lam1_val * q, -50.0, 50.0)
        x2 = np.clip(lam2      * q, -50.0, 50.0)
        Voc = (p1_val * (np.exp(x1) - 1)
             + p2_val * (np.exp(x2) - 1)
             + 4.2)

        Vt   = Voc - v
        disc = Vt**2 - 4 * R0 * P_load
        if disc < 0:
            disc = 0.0
        IL = (Vt - np.sqrt(disc)) / (2 * R0)
        IL = np.clip(IL, 0.0, 100.0)

        VL    = Voc - v - IL * R0
        Ploss = Voc * IL - VL * IL
        Ploss = np.clip(Ploss, -1e3, 1e3)

        dsdt   = -IL / Qeff / 3600.0
        dVrcdt = (IL * R1 - v) / gamma

        return dsdt, dVrcdt, IL, Voc, VL, Ploss

    # 多步RK4积分（从t到t+dt_sim，使用dt_ode作为内部步长）
    def integrate_ode_multistep(soc_in, Vrc_in, Temp_in, P_load, Q_other, dt_total):
        """
        使用细时间步长 dt_ode 从当前状态积分 dt_total 时间
        
        参数:
            soc_in, Vrc_in, Temp_in: 初始状态
            P_load: 恒定负载功率（在dt_total内假设不变）
            Q_other: 恒定热源（在dt_total内假设不变）
            dt_total: 总积分时间（= dt_sim）
        
        返回:
            积分后的状态和物理量
        """
        # 计算需要的ODE步数
        n_ode_steps = int(np.ceil(dt_total / dt_ode))
        dt_actual = dt_total / n_ode_steps  # 实际ODE步长（确保整除）
        
        s_curr = soc_in
        v_curr = Vrc_in
        T_curr = Temp_in
        
        # 定义耦合导数函数
        def deriv_total(s, v, T):
            dsdt, dVrcdt, IL, Voc, VL, Ploss = deriv_ecm(s, v, T, P_load)
            Q_source = W_LOSS * Ploss + Q_other
            Q_source = np.clip(Q_source, -1e3, 1e3)
            dTdt = (Q_source - h_th * (T - Tamb)) / Cth
            dTdt = np.clip(dTdt, -10.0, 10.0)
            return dsdt, dVrcdt, dTdt, IL, Voc, VL, Ploss
        
        # 多步RK4积分
        for _ in range(n_ode_steps):
            # RK4四阶
            ds1, dv1, dT1, _, _, _, _ = deriv_total(s_curr, v_curr, T_curr)
            
            s2 = s_curr + 0.5 * dt_actual * ds1
            v2 = v_curr + 0.5 * dt_actual * dv1
            T2 = np.clip(T_curr + 0.5 * dt_actual * dT1, -50.0, 120.0)
            ds2, dv2, dT2, _, _, _, _ = deriv_total(s2, v2, T2)
            
            s3 = s_curr + 0.5 * dt_actual * ds2
            v3 = v_curr + 0.5 * dt_actual * dv2
            T3 = np.clip(T_curr + 0.5 * dt_actual * dT2, -50.0, 120.0)
            ds3, dv3, dT3, _, _, _, _ = deriv_total(s3, v3, T3)
            
            s4 = s_curr + dt_actual * ds3
            v4 = v_curr + dt_actual * dv3
            T4 = np.clip(T_curr + dt_actual * dT3, -50.0, 120.0)
            ds4, dv4, dT4, _, _, _, _ = deriv_total(s4, v4, T4)
            
            # 更新状态
            s_curr = s_curr + (dt_actual/6.0) * (ds1 + 2*ds2 + 2*ds3 + ds4)
            v_curr = v_curr + (dt_actual/6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
            T_curr = T_curr + (dt_actual/6.0) * (dT1 + 2*dT2 + 2*dT3 + dT4)
            T_curr = np.clip(T_curr, -50.0, 120.0)
        
        # 终点回算物理量
        _, _, IL_end, Voc_end, VL_end, Ploss_end = deriv_ecm(s_curr, v_curr, T_curr, P_load)
        
        return s_curr, v_curr, T_curr, IL_end, Voc_end, VL_end, Ploss_end


    # =========== 主仿真循环 ===========
    print(f"\n[仿真配置] dt_sim={dt_sim}s, dt_ode={dt_ode}s, 抽取因子={output_decimation}")
    print(f"[仿真配置] 总步数={n_steps_sim}, 输出点数={n_output}")
    print(f"[仿真配置] 每个仿真步内ODE子步数={int(dt_sim/dt_ode)}\n")
    
    for i in range(n_steps_sim):
        t = i * dt_sim

        # ── Step 1: 联合泊松过程事件生成 ──
        for app_type in APP_TYPE_KEYS:
            lam_i = lambda_intensity_per_app(t, app_type)
            p_arrive = lam_i * (dt_sim / 3600.0)
            
            if p_arrive <= 0.5:
                n_new = 1 if np.random.rand() < p_arrive else 0
            else:
                n_new = np.random.poisson(p_arrive)
            
            for _ in range(n_new):
                ev = sample_event(t, app_type)
                active_set.append(ev)
                event_log.append(ev)
                arrival_counts[app_type] += 1

        # ── Step 2: 清理已结束事件 ──
        still_active = []
        for ev in active_set:
            if t >= ev['t_end']:
                pass
            else:
                still_active.append(ev)
                app_active_time[ev['app_type']] += dt_sim
        active_set = still_active

        # ── Step 3: 计算当前功耗 ──
        P_events = sum(calc_event_power(ev) for ev in active_set)
        P_load   = P_BASE + P_events
        P_cpu_e, P_scr_e, P_net_e, P_gps_e = calc_module_powers(active_set)

        # ── Step 4: SOC截止检查 ──
        if soc <= 0.05:
            # 截断输出数组
            t_arr       = t_arr[:output_idx]
            soc_arr     = soc_arr[:output_idx]
            Vt_arr      = Vt_arr[:output_idx]
            Voc_arr     = Voc_arr[:output_idx]
            IL_arr      = IL_arr[:output_idx]
            Vrc_arr     = Vrc_arr[:output_idx]
            T_arr       = T_arr[:output_idx]
            P_total_arr = P_total_arr[:output_idx]
            P_loss_arr  = P_loss_arr[:output_idx]
            n_active_arr= n_active_arr[:output_idx]
            P_cpu_arr   = P_cpu_arr[:output_idx]
            P_scr_arr   = P_scr_arr[:output_idx]
            P_net_arr   = P_net_arr[:output_idx]
            P_gps_arr   = P_gps_arr[:output_idx]
            break

        # ── Step 5: 多步RK4积分（dt_sim时间，使用dt_ode子步长） ──
        Q_other = (W_CPU    * P_cpu_e
                 + W_SCREEN * P_scr_e
                 + W_NET    * P_net_e
                 + W_GPS    * P_gps_e)

        soc_new, Vrc_new, T_new, IL_val, Voc_val, VL_val, Ploss_val = \
            integrate_ode_multistep(soc, Vrc, T, P_load, Q_other, dt_sim)
        
        soc_new = max(soc_new, 0.05)

        # ── Step 6: 输出抽取（每output_decimation步记录一次） ──
        if i % output_decimation == 0:
            t_arr[output_idx]       = t
            soc_arr[output_idx]     = soc
            Vrc_arr[output_idx]     = Vrc
            T_arr[output_idx]       = T
            Vt_arr[output_idx]      = VL_val
            Voc_arr[output_idx]     = Voc_val
            IL_arr[output_idx]      = IL_val
            P_total_arr[output_idx] = P_load
            P_loss_arr[output_idx]  = Ploss_val
            n_active_arr[output_idx]= len(active_set)
            P_cpu_arr[output_idx]   = P_cpu_e
            P_scr_arr[output_idx]   = P_scr_e
            P_net_arr[output_idx]   = P_net_e
            P_gps_arr[output_idx]   = P_gps_e
            output_idx += 1

        # ── Step 7: 状态推进 ──
        soc = soc_new
        Vrc = Vrc_new
        T   = T_new
        
        # 进度显示（每10%显示一次）
        if i % (n_steps_sim // 10) == 0:
            progress = 100 * i / n_steps_sim
            print(f"  进度: {progress:.0f}% | SOC={soc:.3f} | T={T:.2f}°C | 活跃事件={len(active_set)}")

    discharge_time_h = t_arr[output_idx-1] / 3600.0 if output_idx > 0 else 0.0

    return {
        't': t_arr[:output_idx], 'soc': soc_arr[:output_idx], 
        'Vt': Vt_arr[:output_idx], 'Voc': Voc_arr[:output_idx],
        'IL': IL_arr[:output_idx], 'Vrc': Vrc_arr[:output_idx], 
        'T': T_arr[:output_idx],
        'P_total': P_total_arr[:output_idx], 'P_loss': P_loss_arr[:output_idx],
        'n_active': n_active_arr[:output_idx],
        'P_cpu': P_cpu_arr[:output_idx], 'P_scr': P_scr_arr[:output_idx],
        'P_net': P_net_arr[:output_idx], 'P_gps': P_gps_arr[:output_idx],
        'event_log': event_log,
        'app_active_time': app_active_time,
        'arrival_counts': arrival_counts,
        'discharge_time_h': discharge_time_h,
        'Qeff': Qeff, 'Q_aging': Q_aging,
        'dt_sim': dt_sim, 'dt_ode': dt_ode,
    }


# ===============================================================
# 运行仿真（细化时间步长）
# ===============================================================
print("=" * 70)
print("运行联合泊松过程仿真（细化时间步长版本）")
print("=" * 70)

# 仿真参数配置
DT_SIM = 0.01            # 仿真主循环步长：1秒
DT_ODE = 0.01            # ODE求解步长：0.1秒（比dt_sim小10倍）
OUTPUT_DECIMATION = 1  # 输出抽取：每10步输出1次（即每10秒输出一次）

all_results = {}
for Q_ag in [0.0]:  # 先只运行新电池，测试性能
    np.random.seed(42)
    Qeff_val = calc_Qeff(Q_ag)
    print(f"\n{'='*70}")
    print(f"Q_aging = {Q_ag:>7.1f} Ah,  Qeff = {Qeff_val:.4f} Ah")
    print(f"{'='*70}")
    
    res = run_joint_poisson_simulation(
        Q_aging=Q_ag, 
        T_sim=86400,              # 24小时
        dt_sim=DT_SIM, 
        dt_ode=DT_ODE,
        output_decimation=OUTPUT_DECIMATION
    )
    all_results[Q_ag] = res
    
    print(f"\n{'='*70}")
    print(f"仿真完成！")
    print(f"  放电时间: {res['discharge_time_h']:.2f} h")
    print(f"  最高温度: {res['T'].max():.2f} °C")
    print(f"  平均负载: {np.mean(res['P_total'])*1000:.1f} mW")
    print(f"  输出数据点: {len(res['t'])}")
    print(f"{'='*70}\n")

# ===============================================================
# 快速可视化（仅新电池）
# ===============================================================
APP_COLORS = {
    'video': '#e74c3c', 'gaming': '#9b59b6', 'social': '#3498db',
    'navigation': '#e67e22', 'calling': '#1abc9c', 'music': '#f39c12',
    'bg_download': '#2ecc71', 'bg_sync': '#95a5a6',
}

r = all_results[0.0]
t_h = r['t'] / 3600.0

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'细化时间步长仿真结果 (dt_sim={DT_SIM}s, dt_ode={DT_ODE}s)', 
             fontsize=14, fontweight='bold')

# SOC
axes[0,0].plot(t_h, r['soc'], linewidth=2, color='#2ecc71')
axes[0,0].axhline(y=0.05, color='red', linestyle=':', linewidth=1.2)
axes[0,0].set_xlabel('时间 (h)', fontsize=11)
axes[0,0].set_ylabel('SOC', fontsize=11)
axes[0,0].set_title('SOC 随时间变化', fontsize=12, fontweight='bold')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_ylim(0, 1.05)

# 温度
axes[0,1].plot(t_h, r['T'], linewidth=2, color='#e74c3c')
axes[0,1].axhline(y=Tamb, color='gray', linestyle='--', linewidth=1)
axes[0,1].set_xlabel('时间 (h)', fontsize=11)
axes[0,1].set_ylabel('温度 (°C)', fontsize=11)
axes[0,1].set_title('电池温度', fontsize=12, fontweight='bold')
axes[0,1].grid(True, alpha=0.3)

# 端电压
axes[1,0].plot(t_h, r['Vt'], linewidth=1.8, color='#3498db')
axes[1,0].set_xlabel('时间 (h)', fontsize=11)
axes[1,0].set_ylabel('端电压 (V)', fontsize=11)
axes[1,0].set_title('负载端电压', fontsize=12, fontweight='bold')
axes[1,0].grid(True, alpha=0.3)

# 放电电流
axes[1,1].plot(t_h, r['IL'], linewidth=1.0, color='#9b59b6', alpha=0.8)
axes[1,1].set_xlabel('时间 (h)', fontsize=11)
axes[1,1].set_ylabel('放电电流 (A)', fontsize=11)
axes[1,1].set_title('放电电流', fontsize=12, fontweight='bold')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("新电池各应用到达统计:")
print("=" * 70)
for app_type in APP_TYPE_KEYS:
    label = APP_TYPES[app_type]['label']
    count = r['arrival_counts'][app_type]
    active_time_h = r['app_active_time'][app_type] / 3600.0
    print(f"  {label:<12}: 到达{count:>4}次, 活跃时长{active_time_h:>6.2f}h")
print("=" * 70)