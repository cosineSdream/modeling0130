
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

def calc_P_screen(B,bs):
    return bs * B * S_screen

def calc_P_cpu(u,Pcpumax, b_cpu):
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
        'label': 'Video',
        'B_mean': 0.72, 'B_std': 0.08,
        'u_mean': 0.22, 'u_std': 0.04,
        'r_mean': 0.70, 'r_std': 0.10,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'dur_mean': 4800.0, 'dur_std': 2400.0,
        # 独立到达率基础强度（相对权重）
        'lambda_weight': 0.22,
    },
    'gaming': {
        'label': 'Gaming',
        'B_mean': 0.85, 'B_std': 0.05,
        'u_mean': 0.65, 'u_std': 0.08,
        'r_mean': 0.45, 'r_std': 0.12,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'dur_mean': 3600.0, 'dur_std': 1800.0,
        'lambda_weight': 0.15,
    },
    'social': {
        'label': 'Social',
        'B_mean': 0.60, 'B_std': 0.10,
        'u_mean': 0.18, 'u_std': 0.05,
        'r_mean': 0.35, 'r_std': 0.08,
        'net_state': '4G_LTE_A', 'gps_state': 'off',
        'dur_mean': 2400.0, 'dur_std': 1200.0,
        'lambda_weight': 0.28,
    },
    'navigation': {
        'label': 'Navigation',
        'B_mean': 0.75, 'B_std': 0.06,
        'u_mean': 0.35, 'u_std': 0.06,
        'r_mean': 0.25, 'r_std': 0.05,
        'net_state': '4G_LTE_A', 'gps_state': 'locating',
        'dur_mean': 1800.0, 'dur_std': 900.0,
        'lambda_weight': 0.10,
    },
    'calling': {
        'label': 'Calling',
        'B_mean': 0.15, 'B_std': 0.05,
        'u_mean': 0.12, 'u_std': 0.03,
        'r_mean': 0.08, 'r_std': 0.02,
        'net_state': '4G_LTE_A', 'gps_state': 'off',
        'dur_mean': 600.0, 'dur_std': 300.0,
        'lambda_weight': 0.12,
    },
    'music': {
        'label': 'Music',
        'B_mean': 0.20, 'B_std': 0.08,
        'u_mean': 0.08, 'u_std': 0.02,
        'r_mean': 0.15, 'r_std': 0.04,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'dur_mean': 7200.0, 'dur_std': 3600.0,
        'lambda_weight': 0.08,
    },
    'bg_download': {
        'label': 'Background Download',
        'B_mean': 0.0, 'B_std': 0.0,
        'u_mean': 0.15, 'u_std': 0.03,
        'r_mean': 0.85, 'r_std': 0.08,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'dur_mean': 1200.0, 'dur_std': 600.0,
        'lambda_weight': 0.09,
    },
    'bg_sync': {
        'label': 'Background Sync',
        'B_mean': 0.0, 'B_std': 0.0,
        'u_mean': 0.05, 'u_std': 0.01,
        'r_mean': 0.20, 'r_std': 0.05,
        'net_state': 'WIFI6', 'gps_state': 'off',
        'dur_mean': 180.0, 'dur_std': 90.0,
        'lambda_weight': 0.09,
    },
}

APP_TYPE_KEYS = list(APP_TYPES.keys())

# 归一化权重，使其和为1
total_weight = sum(APP_TYPES[k]['lambda_weight'] for k in APP_TYPE_KEYS)
for k in APP_TYPE_KEYS:
    APP_TYPES[k]['lambda_weight'] /= total_weight


# ===============================================================
# 联合泊松过程强度函数 λᵢ(t) - 每个应用类型独立
# ===============================================================
def lambda_intensity_per_app(t_sec, app_type):
    """
    每个应用类型的独立到达率函数 λᵢ(t)
    
    参数:
        t_sec: 当前时间（秒）
        app_type: 应用类型键值
    
    返回:
        该应用类型在时间 t 的到达强度（events/hour）
    """
    app = APP_TYPES[app_type]
    t_h = (t_sec % 86400) / 3600.0
    
    # 基础昼夜节律模式（所有应用共享）
    peak1 = 4.0  * np.exp(-0.5 * ((t_h - 13.0) / 3.5)**2)  # 午后高峰
    peak2 = 2.0 * np.exp(-0.5 * ((t_h - 21.0) / 2.5)**2)  # 晚间高峰
    base_intensity = 0.5 + peak1 + peak2
    
    # 应用特定的时间偏好调制
    if app_type == 'video':
        # 视频在晚间更活跃
        time_modulation = 1.0 + 0.5 * np.exp(-0.5 * ((t_h - 20.0) / 3.0)**2)
    elif app_type == 'gaming':
        # 游戏在午后和晚间都活跃
        time_modulation = 1.0 + 0.3 * (np.exp(-0.5 * ((t_h - 15.0) / 2.5)**2) + 
                                        np.exp(-0.5 * ((t_h - 22.0) / 2.0)**2))
    elif app_type == 'social':
        # 社交媒体全天较均匀，早晚略高
        time_modulation = 1.0 + 0.2 * (np.exp(-0.5 * ((t_h - 8.0) / 2.0)**2) + 
                                        np.exp(-0.5 * ((t_h - 19.0) / 2.5)**2))
    elif app_type == 'navigation':
        # 导航在通勤时间高峰
        time_modulation = 1.0 + 0.8 * (np.exp(-0.5 * ((t_h - 8.0) / 1.5)**2) + 
                                        np.exp(-0.5 * ((t_h - 18.0) / 1.5)**2))
    elif app_type == 'calling':
        # 通话在工作时间较多（使用logistic函数平滑过渡）
        # logistic(x) = 1 / (1 + exp(-k*(x - x0)))
        # 早上9点上升，下午18点下降
        morning_factor = 1.0 / (1.0 + np.exp(-3.0 * (t_h - 9.0)))      # 9点处上升
        evening_factor = 1.0 / (1.0 + np.exp(3.0 * (t_h - 18.0)))      # 18点处下降
        time_modulation = 0.8 + 0.5 * morning_factor * evening_factor   # 0.8~1.3之间
    elif app_type == 'music':
        # 音乐播放在通勤和工作时间
        time_modulation = 1.0 + 0.4 * (np.exp(-0.5 * ((t_h - 8.5) / 2.0)**2) + 
                                        np.exp(-0.5 * ((t_h - 14.0) / 3.0)**2))
    elif app_type in ['bg_download', 'bg_sync']:
        # 后台任务相对均匀，夜间略高 —— 使用平滑的 sigmoid 突起替代硬阶跃（在 1~6 点）
        # 使用两个 logistic (sigmoid) 组合形成在区间 [1,6] 的平滑凸起：s(t)=sig(k*(t-1)) - sig(k*(t-6))
        # 可调参数 k_bg 控制过渡陡峭度（越大越接近阶跃）
        k_bg = 6.0
        sig1 = 1.0 / (1.0 + np.exp(-k_bg * (t_h - 1.0)))
        sig2 = 1.0 / (1.0 + np.exp(-k_bg * (t_h - 6.0)))
        bump = sig1 * sig2
        # 原来白天 0.9，夜间 1.2，保持同样振幅（0.3）但平滑过渡
        time_modulation = 0.9 + 0.3 * bump
    else:
        time_modulation = 1.0
    
    # 最终强度 = 基础强度 × 应用权重 × 时间调制
    return base_intensity * app['lambda_weight'] * time_modulation


def lambda_total(t_sec):
    """
    总到达率（所有独立泊松过程的和）
    用于验证和可视化
    """
    return sum(lambda_intensity_per_app(t_sec, app) for app in APP_TYPE_KEYS)


# ===============================================================
# 基础待机功耗
# ===============================================================
def calc_P_base(bs, Pcpumax, b_cpu):
    return calc_P_screen(0.0,bs) + calc_P_cpu(0.02,Pcpumax,b_cpu) + calc_P_net('WIFI6', 0.0) + calc_P_gps('off')

P_BASE = calc_P_base(bs, Pcpumax, b_cpu)
print(f"\n基础待机功耗 P_base = {P_BASE*1000:.2f} mW\n")


# ===============================================================
# 事件功耗贡献（增量，去除 base 部分）
# ===============================================================
def calc_event_power(event, bs, Pcpumax, b_cpu):
    app = APP_TYPES[event['app_type']]
    P_scr     = calc_P_screen(event['B'],bs)
    P_cpu_inc = calc_P_cpu(event['u'],Pcpumax,b_cpu)   - calc_P_cpu(0.02,Pcpumax,b_cpu)
    P_net_inc = calc_P_net(app['net_state'], event['r']) - calc_P_net('WIFI6', 0.0)
    P_gps_inc = calc_P_gps(app['gps_state']) - calc_P_gps('off')
    return max(P_scr + P_cpu_inc + P_net_inc + P_gps_inc, 0.0)


# ===============================================================
# 事件采样（标记 M_k）
# ===============================================================
def sample_event(t_start, app_type):
    """
    为指定应用类型采样一个事件
    
    参数:
        t_start: 事件开始时间
        app_type: 应用类型键值
    """
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
#Cth  = 40.0    # J/K  热容
#Tamb = 25.0    # °C   环境温度
#h_th = 20.0*0.0128    # W/K  散热系数

# 热贡献权重系数
W_LOSS   = 1.0
W_CPU    = 1.0
W_SCREEN = 0.3
W_NET    = 0.7
W_GPS    = 0.7

# 老化量列表 (Ah)
Q_AGING_LIST = [0,100,500,1000,5000]#[0.0, 100, 500, 1000, 5000]

# Qeff 老化衰减比率
QEFF_DECAY_RATIO = 1.314e-3
QEFF_NOMINAL     = 5.0


# ===============================================================
# 温度/老化相关的 OCV 参数函数
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
    return max(QEFF_NOMINAL*( 1 - 4*QEFF_DECAY_RATIO * np.sqrt(Q_aging)), 0.5)


# ===============================================================
# 分解各模块功耗
# ===============================================================
def calc_module_powers(active_set,bs, Pcpumax, b_cpu):   
    P_cpu  = 0.0
    P_scr  = 0.0
    P_net  = 0.0
    P_gps  = 0.0
    for ev in active_set:
        app = APP_TYPES[ev['app_type']]
        P_scr  += calc_P_screen(ev['B'],bs)
        P_cpu  += calc_P_cpu(ev['u'],Pcpumax,b_cpu) - calc_P_cpu(0.02,Pcpumax,b_cpu)
        P_net  += calc_P_net(app['net_state'], ev['r']) - calc_P_net('WIFI6', 0.0)
        P_gps  += calc_P_gps(app['gps_state']) - calc_P_gps('off')
    P_cpu  += calc_P_cpu(0.02,Pcpumax,b_cpu)
    P_scr  += calc_P_screen(0.0,bs)
    P_net  += calc_P_net('WIFI6', 0.0)
    P_gps  += calc_P_gps('off')
    return P_cpu, P_scr, P_net, P_gps


# ===============================================================
# 核心仿真函数（联合泊松过程版本）
# ===============================================================
def run_joint_poisson_simulation(Q_aging, T_sim, dt, bs, Pcpumax, b_cpu, Cth, Tamb, h_th):
    """
    联合泊松过程 + Thevenin ECM + 一阶热模型 耦合仿真
    
    核心改进：
    1. 每个应用类型有独立的泊松过程
    2. 各应用的到达率 λᵢ(t) 相互独立
    3. 在每个时间步中，为每个应用类型独立检测事件到达
    
    参数:
        Q_aging: 电池老化量 (Ah)
        T_sim: 仿真总时长（秒）
        dt: 时间步长（秒）
    """
    Qeff   = calc_Qeff(Q_aging)
    #tau_th = Cth / h_th
    lam2   = calc_lambda2(Q_aging)
    #print(Cth)
    # 时间序列初始化
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
    
    # 每个应用类型的到达次数统计
    arrival_counts = {k: 0 for k in APP_TYPE_KEYS}

    # 状态初始化
    soc = 1.0
    Vrc = 0.0
    T   = Tamb

    # 事件管理
    active_set      = []
    event_log       = []
    app_active_time = {k: 0.0 for k in APP_TYPES}

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

    # RK4 积分
    def rk4_step_ecm(soc_in, Vrc_in, Temp_in, P_load, Q_other, h_step,Cth,Tamb,h_th):
        def deriv_total(s, v, T, P_load = P_load, Q_other=Q_other, Cth=Cth, Tamb=Tamb, h_th=h_th):
            dsdt, dVrcdt, IL, Voc, VL, Ploss = deriv_ecm(s, v, T, P_load)
            Q_source = W_LOSS * Ploss + Q_other
            Q_source = np.clip(Q_source, -1e3, 1e3)
            dTdt = (Q_source - h_th * (T - Tamb)) / Cth
            dTdt = np.clip(dTdt, -10.0, 10.0)
            return dsdt, dVrcdt, dTdt, IL, Voc, VL, Ploss

        ds1, dv1, dT1, _, _, _, _ = deriv_total(soc_in, Vrc_in, Temp_in, P_load, Q_other, Cth, Tamb, h_th)

        s2 = soc_in  + 0.5 * h_step * ds1
        v2 = Vrc_in  + 0.5 * h_step * dv1
        T2 = np.clip(Temp_in + 0.5 * h_step * dT1, -50.0, 120.0)
        ds2, dv2, dT2, _, _, _, _ = deriv_total(s2, v2, T2, P_load, Q_other, Cth, Tamb, h_th)

        s3 = soc_in  + 0.5 * h_step * ds2
        v3 = Vrc_in  + 0.5 * h_step * dv2
        T3 = np.clip(Temp_in + 0.5 * h_step * dT2, -50.0, 120.0)
        ds3, dv3, dT3, _, _, _, _ = deriv_total(s3, v3, T3, P_load, Q_other, Cth, Tamb, h_th)

        s4 = soc_in  + h_step * ds3
        v4 = Vrc_in  + h_step * dv3
        T4 = np.clip(Temp_in + h_step * dT3, -50.0, 120.0)
        ds4, dv4, dT4, _, _, _, _ = deriv_total(s4, v4, T4, P_load, Q_other, Cth, Tamb, h_th)

        soc_out  = soc_in + (h_step/6.0) * (ds1 + 2*ds2 + 2*ds3 + ds4)
        Vrc_out  = Vrc_in + (h_step/6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        Temp_out = Temp_in + (h_step/6.0) * (dT1 + 2*dT2 + 2*dT3 + dT4)

        _, _, IL_end, Voc_end, VL_end, Ploss_end = deriv_ecm(soc_out, Vrc_out, Temp_out, P_load)
        return soc_out, Vrc_out, Temp_out, IL_end, Voc_end, VL_end, Ploss_end


    # =========== 主仿真循环（联合泊松过程） ===========
    for i in range(n_steps):
        t = i * dt
        t_arr[i] = t

        # ── Step 1: 联合泊松过程 - 为每个应用类型独立检测事件到达 ──
        for app_type in APP_TYPE_KEYS:
            lam_i = lambda_intensity_per_app(t, app_type)
            p_arrive = lam_i * (dt / 3600.0)  # 转换为时间步内的概率
            
            # 泊松到达检测
            if p_arrive <= 0.5:
                n_new = 1 if np.random.rand() < p_arrive else 0
            else:
                n_new = np.random.poisson(p_arrive)
            
            # 为该应用类型生成新事件
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
                app_active_time[ev['app_type']] += dt
        active_set = still_active

        # ── Step 3: 计算总负载和各模块功耗 ──
        P_events = sum(calc_event_power(ev,bs,Pcpumax,b_cpu) for ev in active_set)
        P_load   = P_BASE + P_events
        P_cpu_e, P_scr_e, P_net_e, P_gps_e = calc_module_powers(active_set,bs,Pcpumax,b_cpu)

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

        # ── Step 5: RK4求解耦合态 ──
        Q_other = (W_CPU    * P_cpu_e
                 + W_SCREEN * P_scr_e
                 + W_NET    * P_net_e
                 + W_GPS    * P_gps_e)

        soc_new, Vrc_new, T_new, IL_val, Voc_val, VL_val, Ploss_val = \
            rk4_step_ecm(soc, Vrc, T, P_load, Q_other, dt, Cth, Tamb, h_th)
        #print(Cth)
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
        'arrival_counts': arrival_counts,
        'discharge_time_h': discharge_time_h,
        'Qeff': Qeff, 'Q_aging': Q_aging,
    }

'''
# ===============================================================
# 运行所有老化水平的仿真
# ===============================================================
print("=" * 70)
print("运行联合泊松过程多老化水平仿真 ...")
print("=" * 70)

all_results = {}
for Q_ag in Q_AGING_LIST:
    np.random.seed(42)
    Qeff_val = calc_Qeff(Q_ag)
    print(f"\n  Q_aging = {Q_ag:>7.1f} Ah,  Qeff = {Qeff_val:.4f} Ah")
    res = run_joint_poisson_simulation(Q_aging=Q_ag, T_sim=86400, dt=10.0)
    all_results[Q_ag] = res
    print(f"    放电时间: {res['discharge_time_h']:.2f} h, "
          f"最高温度: {res['T'].max():.2f} °C, "
          f"平均负载: {np.mean(res['P_total'])*1000:.1f} mW")
    print(f"    各应用到达次数: {res['arrival_counts']}")

print("\n" + "=" * 70)


# ===============================================================
# 颜色映射
# ===============================================================
APP_COLORS = {
    'video': '#e74c3c', 'gaming': '#9b59b6', 'social': '#3498db',
    'navigation': '#e67e22', 'calling': '#1abc9c', 'music': '#f39c12',
    'bg_download': '#2ecc71', 'bg_sync': '#95a5a6',
}

AGING_COLORS = {
    0.0:    '#2ecc71',
    100.0:   '#3498db',
    500.0:   '#f39c12',
    1000.0:  '#e67e22',
    5000.0: '#e74c3c',
}

AGING_LABELS = {
    0.0:    'New battery (Q=0)',
    100.0: 'Light aging (Q=100)',
    500.0:   'Moderate aging (Q=500)',
    1000.0:  'Severe aging (Q=1000)',
    5000.0: 'Extreme aging (Q=5000)',
}


# ===============================================================
# 图 1: 联合泊松过程 - 各应用独立强度函数 λᵢ(t)
# ===============================================================
fig1, axes1 = plt.subplots(2, 1, figsize=(14, 10))
fig1.suptitle('Joint Poisson Process: Per-app arrival intensity functions', fontsize=15, fontweight='bold')

t_24h = np.linspace(0, 86400, 1000)
t_h_arr = t_24h / 3600.0

# (0) 各应用独立强度
for app_type in APP_TYPE_KEYS:
    lam_vals = [lambda_intensity_per_app(t, app_type) for t in t_24h]
    axes1[0].plot(t_h_arr, lam_vals, linewidth=2.0, 
                  color=APP_COLORS[app_type], label=APP_TYPES[app_type]['label'])

axes1[0].set_xlabel('Time (h)', fontsize=11)
axes1[0].set_ylabel('Arrival intensity λi(t) (events/h)', fontsize=11)
axes1[0].set_title('Per-app arrival intensity', fontsize=12, fontweight='bold')
axes1[0].set_xticks(range(0, 25, 2))
axes1[0].legend(fontsize=9, loc='upper left', ncol=2)
axes1[0].grid(True, alpha=0.3)

# (1) 总到达率（所有应用的和）
lam_total_vals = [lambda_total(t) for t in t_24h]
axes1[1].fill_between(t_h_arr, lam_total_vals, alpha=0.3, color='#3498db')
axes1[1].plot(t_h_arr, lam_total_vals, linewidth=2.5, color='#2980b9')
axes1[1].set_xlabel('Time (h)', fontsize=11)
axes1[1].set_ylabel('Total arrival intensity Σλi(t) (events/h)', fontsize=11)
axes1[1].set_title('Total arrival rate (sum over apps)', fontsize=12, fontweight='bold')
axes1[1].set_xticks(range(0, 25, 2))
axes1[1].grid(True, alpha=0.3)
axes1[1].axvspan(0, 7, alpha=0.08, color='gray', label='Late night/early morning')
axes1[1].axvspan(22, 24, alpha=0.08, color='gray')
axes1[1].legend(fontsize=10)

plt.tight_layout()
plt.show()


# ===============================================================
# 图 2: 多老化水平对比 — SOC / 温度 / 端电压 / 放电电流
# ===============================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.suptitle('Battery state and temperature evolution across aging levels (Joint Poisson Process)', fontsize=15, fontweight='bold')

for Q_ag in Q_AGING_LIST:
    r     = all_results[Q_ag]
    t_h   = r['t'] / 3600.0
    color = AGING_COLORS[Q_ag]
    label = AGING_LABELS[Q_ag]

    axes2[0,0].plot(t_h, r['soc'], linewidth=2, color=color, label=label)
    axes2[0,1].plot(t_h, r['T'],   linewidth=2, color=color, label=label)
    axes2[1,0].plot(t_h, r['Vt'],  linewidth=1.8, color=color, label=label)
    axes2[1,1].plot(t_h, r['IL'],  linewidth=1.0, color=color, alpha=0.8, label=label)

axes2[0,0].axhline(y=0.05, color='red', linestyle=':', linewidth=1.2, alpha=0.6)
axes2[0,0].set_xlabel('Time (h)', fontsize=11)
axes2[0,0].set_ylabel('SOC', fontsize=11)
axes2[0,0].set_title('SOC over time', fontsize=12, fontweight='bold')
axes2[0,0].legend(fontsize=8, loc='best')
axes2[0,0].grid(True, alpha=0.3)
axes2[0,0].set_ylim(0, 1.05)

axes2[0,1].axhline(y=Tamb, color='gray', linestyle='--', linewidth=1, label=f'Ambient temp {Tamb}°C')
axes2[0,1].set_xlabel('Time (h)', fontsize=11)
axes2[0,1].set_ylabel('Temperature T (°C)', fontsize=11)
axes2[0,1].set_title('Battery temperature over time', fontsize=12, fontweight='bold')
axes2[0,1].legend(fontsize=8, loc='best')
axes2[0,1].grid(True, alpha=0.3)

axes2[1,0].set_xlabel('Time (h)', fontsize=11)
axes2[1,0].set_ylabel('Load terminal voltage VL (V)', fontsize=11)
axes2[1,0].set_title('Terminal voltage over time', fontsize=12, fontweight='bold')
axes2[1,0].legend(fontsize=8, loc='best')
axes2[1,0].grid(True, alpha=0.3)

axes2[1,1].set_xlabel('Time (h)', fontsize=11)
axes2[1,1].set_ylabel('Discharge current IL (A)', fontsize=11)
axes2[1,1].set_title('Discharge current over time', fontsize=12, fontweight='bold')
axes2[1,1].legend(fontsize=8, loc='best')
axes2[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ===============================================================
# 图 3: Gantt 图 + 功耗叠加（新电池，展示联合泊松效果）
# ===============================================================
r_ref = all_results[0.0]
t_end_sim = r_ref['t'][-1] if len(r_ref['t']) > 0 else 86400

fig3, (ax_gantt, ax_power) = plt.subplots(2, 1, figsize=(16, 9),
                                            gridspec_kw={'height_ratios': [2, 1]})
fig3.suptitle('Joint Poisson Process: Parallel event timelines and total power (New battery)', fontsize=15, fontweight='bold')

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
ax_gantt.set_xlabel('Time (h)', fontsize=11)
ax_gantt.set_title('Event timelines by app (Gantt chart)', fontsize=12)
ax_gantt.grid(axis='x', alpha=0.3)
ax_gantt.set_xlim(0, t_end_sim / 3600.0)
legend_patches = [mpatches.Patch(color=APP_COLORS[k], label=APP_TYPES[k]['label'])
                  for k in type_keys_ordered]
ax_gantt.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)

t_h = r_ref['t'] / 3600.0
ax_power.plot(t_h, r_ref['P_total']*1000, linewidth=1.0, color='#e74c3c', alpha=0.7)
ax_power.fill_between(t_h, P_BASE*1000, r_ref['P_total']*1000,
                       alpha=0.25, color='#e74c3c', label='Event power (stacked)')
ax_power.axhline(y=P_BASE*1000, color='#2c3e50', linestyle='--',
                 linewidth=1.5, label=f'Base standby {P_BASE*1000:.1f} mW')
ax_power.set_xlabel('Time (h)', fontsize=11)
ax_power.set_ylabel('Total load P(t) (mW)', fontsize=11)
ax_power.set_title('Total power over time', fontsize=12)
ax_power.legend(fontsize=9, loc='upper left')
ax_power.grid(True, alpha=0.3)
ax_power.set_xlim(0, t_end_sim / 3600.0)

ax_n = ax_power.twinx()
ax_n.plot(t_h, r_ref['n_active'], linewidth=1.2, color='#27ae60', alpha=0.8)
ax_n.set_ylabel('Active events count', fontsize=11, color='#27ae60')
ax_n.tick_params(axis='y', labelcolor='#27ae60')
ax_n.set_ylim(0, max(r_ref['n_active'].max()+1, 6))

plt.tight_layout()
plt.show()


# ===============================================================
# 图 4: 联合泊松过程特有 - 各应用到达次数分布
# ===============================================================
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle('Joint Poisson Process: Event arrival statistics by app', fontsize=15, fontweight='bold')

r_new = all_results[0.0]

# (0) 各应用到达次数柱状图
app_labels = [APP_TYPES[k]['label'] for k in APP_TYPE_KEYS]
app_colors_list = [APP_COLORS[k] for k in APP_TYPE_KEYS]
arrival_nums = [r_new['arrival_counts'][k] for k in APP_TYPE_KEYS]

bars = axes4[0].bar(range(len(APP_TYPE_KEYS)), arrival_nums, 
                     color=app_colors_list, edgecolor='#2c3e50', linewidth=0.7)
for i, v in enumerate(arrival_nums):
    axes4[0].text(i, v + 0.5, str(v), ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
axes4[0].set_xticks(range(len(APP_TYPE_KEYS)))
axes4[0].set_xticklabels(app_labels, fontsize=9, rotation=30, ha='right')
axes4[0].set_ylabel('Arrival counts', fontsize=11)
axes4[0].set_title('Event arrival counts by app (24h)', fontsize=12, fontweight='bold')
axes4[0].grid(axis='y', alpha=0.3)

# (1) 各应用活跃时间占比饼图
active_times = [r_new['app_active_time'][k] for k in APP_TYPE_KEYS]
total_active = sum(active_times)
if total_active > 0:
    percentages = [t/total_active*100 for t in active_times]
    # 只显示活跃时间 > 0 的应用标签，其他设为空字符串
    pie_labels = [app_labels[i] if active_times[i] > 0 else '' for i in range(len(APP_TYPE_KEYS))]
    wedges, texts, autotexts = axes4[1].pie(percentages, labels=pie_labels, 
                                              colors=app_colors_list,
                                              autopct='%1.1f%%', startangle=90,
                                              textprops={'fontsize': 9})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
axes4[1].set_title('Active time share by app', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


# ===============================================================
# 图 5: 老化水平对比汇总柱状图
# ===============================================================
fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5.5))
fig5.suptitle('Summary across aging levels (Joint Poisson Process)', fontsize=14, fontweight='bold')

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
axes5[0].set_ylabel('Discharge time (h)', fontsize=11)
axes5[0].set_title('Lifespan by aging level', fontsize=12, fontweight='bold')
axes5[0].grid(axis='y', alpha=0.3)

# (1) 最高温度
bars = axes5[1].bar(range(len(Q_AGING_LIST)), max_temps, color=Q_colors,
                    edgecolor='#2c3e50', linewidth=0.7)
for i, v in enumerate(max_temps):
    axes5[1].text(i, v + 0.05, f'{v:.2f}°C', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes5[1].set_xticks(range(len(Q_AGING_LIST)))
axes5[1].set_xticklabels(Q_labels, fontsize=8, rotation=15)
axes5[1].set_ylabel('Max temperature (°C)', fontsize=11)
axes5[1].set_title('Peak temperature by aging level', fontsize=12, fontweight='bold')
axes5[1].axhline(y=Tamb, color='gray', linestyle=':', linewidth=1)
axes5[1].grid(axis='y', alpha=0.3)

# (2) 有效容量 Qeff
bars = axes5[2].bar(range(len(Q_AGING_LIST)), Qeff_vals, color=Q_colors,
                    edgecolor='#2c3e50', linewidth=0.7)
for i, v in enumerate(Qeff_vals):
    axes5[2].text(i, v + 0.02, f'{v:.3f}Ah', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes5[2].set_xticks(range(len(Q_AGING_LIST)))
axes5[2].set_xticklabels(Q_labels, fontsize=8, rotation=15)
axes5[2].set_ylabel('Effective capacity Qeff (Ah)', fontsize=11)
axes5[2].set_title('Effective capacity by aging level', fontsize=12, fontweight='bold')
axes5[2].axhline(y=QEFF_NOMINAL, color='gray', linestyle=':', linewidth=1, label=f'New battery {QEFF_NOMINAL} Ah')
axes5[2].legend(fontsize=9)
axes5[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# ===============================================================
# 最终汇总输出
# ===============================================================
print("\n" + "=" * 70)
print("Summary of joint Poisson process simulations across aging levels")
print("=" * 70)
print(f"{'Aging Level':<22} {'Qeff(Ah)':>9} {'Discharge(h)':>12} {'Avg Load(mW)':>13} "
    f"{'Max Temp(°C)':>13} {'Max Parallel':>12} {'Total Events':>13}")
print("-" * 95)
for Q_ag in Q_AGING_LIST:
    r = all_results[Q_ag]
    print(f"{AGING_LABELS[Q_ag]:<22} "
          f"{calc_Qeff(Q_ag):>8.4f} "
          f"{r['discharge_time_h']:>11.2f} "
          f"{np.mean(r['P_total'])*1000:>12.1f} "
          f"{r['T'].max():>12.2f} "
          f"{int(r['n_active'].max()):>12} "
          f"{len(r['event_log']):>13}")
print("=" * 95)

print("\n" + "=" * 70)
print("New battery arrival statistics:")
print("=" * 70)
r_new = all_results[0.0]
for app_type in APP_TYPE_KEYS:
    label = APP_TYPES[app_type]['label']
    count = r_new['arrival_counts'][app_type]
    active_time_h = r_new['app_active_time'][app_type] / 3600.0
    print(f"  {label:<18}: arrivals {count:>4}, active time {active_time_h:>6.2f} h")
print("=" * 70)
# ===============================================================
# 运行不同环境温度下的仿真
# ===============================================================
print("=" * 70)
print("运行不同环境温度下的联合泊松过程仿真 ...")
print("=" * 70)

# 环境温度列表
Tamb_list = [-200, -40, 25, 60]
Idk = [-25 , 0, 25, 35]
# 存储不同环境温度下的仿真结果
all_results_by_temp = {}

# 仿真
for Tamb in Tamb_list:
    print(f"\n 环境温度 Tamb = {Tamb} °C")
    
    # 运行所有老化水平的仿真
    for Q_ag in Q_AGING_LIST:
        np.random.seed(42)
        Qeff_val = calc_Qeff(Q_ag)
        print(f"  Q_aging = {Q_ag:>7.1f} Ah,  Qeff = {Qeff_val:.4f} Ah")
        res = run_joint_poisson_simulation(Q_aging=Q_ag, T_sim=86400, dt=10.0)
        all_results_by_temp[(Tamb, Q_ag)] = res
        print(f"    放电时间: {res['discharge_time_h']:.2f} h, "
              f"最高温度: {res['T'].max():.2f} °C, "
              f"平均负载: {np.mean(res['P_total'])*1000:.1f} mW")
        print(f"    各应用到达次数: {res['arrival_counts']}")

print("\n" + "=" * 70)

# ===============================================================
# 绘制不同环境温度下的SOC随时间变化曲线
# ===============================================================
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle('SOC Over Time at Different Ambient Temperatures', fontsize=16, fontweight='bold')

# 创建颜色映射 (这里用`viridis`色图，您可以选择其他色图)
cmap = plt.get_cmap('viridis')  # 选择合适的颜色映射
norm = plt.Normalize(vmin=min(Tamb_list), vmax=max(Tamb_list))  # 归一化温度范围

for i, Tamb in enumerate(Tamb_list):
    # 只绘制老化水平 Q_aging = 0 时的曲线
    res = all_results_by_temp[(Tamb, 0)]  # 只取 Q_aging = 0 的仿真结果
    t_h = res['t'] / 3600.0
    color = cmap(norm(Tamb))  # 根据温度获取颜色
    label = f"T={Idk[i]}°C"  # 标签只包含环境温度
    ax.plot(t_h, res['soc'], color=color, label=label, linewidth=2)

# 设置图表
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('State of Charge (SOC)', fontsize=12)
ax.set_title('SOC Over Time for Different Ambient Temperatures (Q_aging = 0)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right', ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 假设你已经有了定义好的函数，比如run_joint_poisson_simulation，Q_aging的计算，等
# 这里的函数会被用来在不同的参数设置下进行仿真，并计算“time to empty”。

# 假设的初始条件
soc_initial = 1.0
T_sim = 86400  # 仿真时长 (秒)，1天
dt = 10.0  # 时间步长

# 定义用于敏感性分析的可变参数
params = {
    'bs': [2.464, 2.464*1.05, 2.464*0.95],  # 屏幕亮度常数
    'Pcpumax': [4.0, 4.0*1.05, 4.0*0.95],  # 最大CPU功率
    'b_cpu': [0.1, 0.1*1.05, 0.1*0.95],  # CPU功率系数
    'Cth': [40.0, 40.0*1.05, 40.0*0.95],  # 热容
    #'Tamb': [25.0, 25.0*1.5, 25.0*0.5],  # 环境温度
    'h_th': [20*0.0128,20*0.0128*1.05, 20*0.0128*0.95],  # 散热系数
}

# 用来计算“time to empty”的辅助函数
def calculate_time_to_empty(bs, Pcpumax, b_cpu, Cth, Tamb, h_th):
    """
    使用run_joint_poisson_simulation来计算时间直到电池耗尽
    """
    
    result = run_joint_poisson_simulation(Q_aging=0, T_sim=T_sim, dt=dt, bs=bs, Pcpumax=Pcpumax, b_cpu=b_cpu, Cth=Cth, Tamb=Tamb, h_th=h_th)
    return result['discharge_time_h']  # 返回耗尽时间（小时）

# 进行敏感性分析
def perform_sensitivity_analysis():
    """
    进行单因子敏感性分析，并生成龙卷风图
    """
    # 存储每个参数对time to empty的影响
    sensitivity_results = []

    # 计算基准量（使用原始参数值）
    base_time_to_empty = calculate_time_to_empty(
        bs=2.464,  # 屏幕亮度
        Pcpumax=4.0,  # 最大CPU功率
        b_cpu=0.1,  # CPU功率系数
        Cth=40.0,  # 热容
        Tamb=25.0,  # 环境温度
        h_th=20*0.0128  # 散热系数
    )
    
    # 对每个参数进行敏感性分析
    for param_name, param_values in params.items():
        time_to_empty_results = []
          # 保持随机种子一致以减少随机性影响
        
        for value in param_values:
            np.random.seed(42)
            # 修改一个参数值并进行仿真
            # 注意：其它参数保持初始值
            modified_params = {
                'bs': 2.464,  # 使用默认值
                'Pcpumax': 4.0,  # 使用默认值
                'b_cpu': 0.1,  # 使用默认值
                'Cth': 40.0,  # 使用默认值
                #'Tamb': 25.0,  # 使用默认值
                'h_th': 20*0.0128,  # 使用默认值
            }
            modified_params[param_name] = value
            
            # 运行仿真并计算“time to empty”
            time_to_empty = calculate_time_to_empty(
                bs=modified_params['bs'],
                Pcpumax=modified_params['Pcpumax'],
                b_cpu=modified_params['b_cpu'],
                Cth=modified_params['Cth'],
                Tamb=25.0,  # 使用默认值
                h_th=modified_params['h_th']
            )
            # 计算差值（time to empty 减去基准量）
            time_to_empty_diff = time_to_empty - base_time_to_empty
            time_to_empty_results.append(time_to_empty_diff)
        
        sensitivity_results.append((param_name, param_values, time_to_empty_results))
    
    # 创建龙卷风图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 横轴：time to empty的变化
    for i, (param_name, param_values, time_to_empty_results) in enumerate(sensitivity_results):
        bars = ax.barh(
            [f"{param_name}: {v:.2f}" for v in param_values], 
            time_to_empty_results,
            label=param_name
        )

        # Adding value labels to the bars
        for bar in bars:
            ax.text(
                bar.get_width(),  # x position of label (end of the bar)
                bar.get_y() + bar.get_height() / 2,  # y position (center of the bar)
                f'{bar.get_width():.2e}',  # Format the value to 2 decimal places
                va='center',  # Vertically center the text
                ha='left',  # Align text to the left
                color='black'  # Text color
            )

    ax.set_xlabel('Change in Time to Empty (hours)')
    ax.set_title('Sensitivity Analysis of Battery Life (Difference from Base)')
    ax.legend(title="Parameters", loc='best')
    plt.tight_layout()
    plt.show()
    # 计算基准量（使用原始参数值）
    base_time_to_empty = calculate_time_to_empty(
        bs=2.464,  # 屏幕亮度
        Pcpumax=4.0,  # 最大CPU功率
        b_cpu=0.1,  # CPU功率系数
        Cth=40.0,  # 热容
        Tamb=25.0,  # 环境温度
        h_th=20*0.0128  # 散热系数
    )


# 执行敏感性分析
perform_sensitivity_analysis()



