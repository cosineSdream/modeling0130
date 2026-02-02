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
# Sigmoid平滑函数（用于平滑变量约束）
# ===============================
def sigmoid(x, k=1.0):
    """
    标准sigmoid函数: 1 / (1 + exp(-kx))
    用于将任意实数映射到 (0, 1) 区间
    """
    x_clip = np.clip(k * x, -100, 100)
    return 1.0 / (1.0 + np.exp(-x_clip))

def sigmoid_range(x, x_min, x_max, k=1.0):
    """
    将sigmoid拓展到任意区间 [x_min, x_max]
    x_range = x_max - x_min
    output ≈ x_min + x_range * sigmoid((x - center) / scale)
    """
    center = (x_min + x_max) / 2.0
    scale = (x_max - x_min) / 6.0  # 从-3到+3 sigma覆盖约99.7%范围
    return x_min + (x_max - x_min) * sigmoid((x - center) / scale, k)

def soft_clip(x, x_min, x_max, k=3.0):
    """
    使用sigmoid进行软clipping（平滑的硬边界替代）
    替代 np.clip(x, x_min, x_max)
    """
    # 下界约束：如果 x < x_min，使用sigmoid平滑上升
    # 上界约束：如果 x > x_max，使用sigmoid平滑下降
    lower_sigmoid = sigmoid_range(x, x_min, (x_min + x_max) / 2.0, k)
    upper_sigmoid = sigmoid_range(x, (x_min + x_max) / 2.0, x_max, k)
    # 综合约束
    return x_min + (x_max - x_min) * sigmoid((x - x_min) / (x_max - x_min + 1e-8), k)

def smooth_transition(x, lower, upper, k=2.0):
    """
    从 lower 到 upper 的平滑过渡函数
    类似于 logistic(x, k, (lower+upper)/2) 缩放到 [lower, upper]
    """
    midpoint = (lower + upper) / 2.0
    width = (upper - lower) / 2.0
    return midpoint + width * np.tanh(k * (x - midpoint) / width)

def clipped_variable(x, x_min, x_max, k=2.0, method='sigmoid'):
    """
    对变量进行平滑约束
    方法: 'sigmoid' - 使用sigmoid平滑, 'clip' - 使用硬边界, 'tanh' - 使用tanh
    """
    if method == 'sigmoid':
        return sigmoid_range(x, x_min, x_max, k)
    elif method == 'tanh':
        return smooth_transition(x, x_min, x_max, k)
    else:  # 'clip'
        return np.clip(x, x_min, x_max)

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
print(f"R0 R2={r_squared(R0_table, R0_fit):.6f}, RMSE={rmse(R0_table, R0_fit)*1000:.4f} mOhm")
print(f"R1 R2={r_squared(R1_table, R1_fit):.6f}, RMSE={rmse(R1_table, R1_fit)*1000:.4f} mOhm")
print(f"C1 R2={r_squared(C1_table, C1_fit):.6f}, RMSE={rmse(C1_table, C1_fit):.4f} F")
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
        # 独立到达率基础强度（相对权重）
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
    peak1 = 8.0  * np.exp(-0.5 * ((t_h - 13.0) / 3.5)**2)  # 午后高峰
    peak2 = 10.0 * np.exp(-0.5 * ((t_h - 21.0) / 2.5)**2)  # 晚间高峰
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
        # 后台任务相对均匀，夜间略高
        if 1 <= t_h <= 6:
            time_modulation = 1.2
        else:
            time_modulation = 0.9
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
Cth  = 40.0    # J/K  热容
Tamb = 25.0    # °C   环境温度
h_th = 20.0    # W/K  散热系数

# 热贡献权重系数
W_LOSS   = 1.0
W_CPU    = 1.0
W_SCREEN = 0.3
W_NET    = 0.7
W_GPS    = 0.7

# 老化量列表 (Ah)
Q_AGING_LIST = [0.0, 100, 500, 1000, 5000]

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
    return max(QEFF_NOMINAL*( 1 - QEFF_DECAY_RATIO * np.sqrt(Q_aging)), 0.5)


# ===============================================================
# 分解各模块功耗（使用sigmoid平滑过渡）
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
    
    # 对功率使用软约束确保物理合理性
    P_cpu = np.clip(P_cpu, 0.0, 5.0)
    P_scr = np.clip(P_scr, 0.0, 2.0)
    P_net = np.clip(P_net, 0.0, 1.0)
    P_gps = np.clip(P_gps, 0.0, 1.0)
    
    return P_cpu, P_scr, P_net, P_gps


# ===============================================================
# 核心仿真函数（联合泊松过程版本）
# ===============================================================
def run_joint_poisson_simulation(Q_aging, T_sim=86400, dt=10.0, dt_ode=1.0):
    """
    联合泊松过程 + Thevenin ECM + 一阶热模型 耦合仿真
    
    核心改进：
    1. 每个应用类型有独立的泊松过程
    2. 各应用的到达率 λᵢ(t) 相互独立
    3. 在每个时间步中，为每个应用类型独立检测事件到达
    4. 使用自适应ODE步长和sigmoid平滑约束
    
    参数:
        Q_aging: 电池老化量 (Ah)
        T_sim: 仿真总时长（秒）
        dt: 主时间步长（秒）
        dt_ode: ODE积分步长（秒），越小精度越高，建议1.0-2.0
    """
    Qeff   = calc_Qeff(Q_aging)
    tau_th = Cth / h_th
    lam2   = calc_lambda2(Q_aging)

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
        # 使用soft_clip进行平滑SOC约束，避免硬边界
        s_c   = soft_clip(s, 0.05, 1.0, k=3.0)
        R0    = max(float(R0_poly(s_c)), 1e-6)
        R1    = max(float(R1_poly(s_c)), 1e-6)
        C1    = max(float(C1_poly(s_c)), 1e-6)
        gamma = R1 * C1

        q = (1 - s_c) * Qeff

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
        # 使用soft_clip平滑限制电流，替代硬边界
        IL = soft_clip(IL, 0.0, 100.0, k=2.0)

        VL    = Voc - v - IL * R0
        Ploss = Voc * IL - VL * IL
        # 使用平滑约束限制功耗变化范围
        Ploss = np.clip(Ploss, -1e3, 1e3)

        dsdt   = -IL / Qeff / 3600.0
        dVrcdt = (IL * R1 - v) / gamma

        return dsdt, dVrcdt, IL, Voc, VL, Ploss

    # RK4 积分（带自适应子步长）
    def rk4_step_ecm(soc_in, Vrc_in, Temp_in, P_load, Q_other, h_step):
        """
        使用RK4积分一个 h_step 的时间间隔
        内部使用更小的 dt_ode 步长进行多次积分以提高精度
        """
        def deriv_total(s, v, T):
            dsdt, dVrcdt, IL, Voc, VL, Ploss = deriv_ecm(s, v, T, P_load)
            Q_source = W_LOSS * Ploss + Q_other
            # 对热源进行软约束，确保温度变化平滑
            Q_source = np.clip(Q_source, -1e3, 1e3)
            # 温度导数计算
            dTdt = (Q_source - h_th * (T - Tamb)) / Cth
            dTdt = np.clip(dTdt, -10.0, 10.0)
            return dsdt, dVrcdt, dTdt, IL, Voc, VL, Ploss

        # 在h_step内进行多个小步长的RK4积分
        n_substeps = max(1, int(np.ceil(h_step / dt_ode)))
        h_sub = h_step / n_substeps
        
        soc_current  = soc_in
        vrc_current  = Vrc_in
        temp_current = Temp_in
        IL_end, Voc_end, VL_end, Ploss_end = 0.0, 0.0, 0.0, 0.0
        
        for _ in range(n_substeps):
            ds1, dv1, dT1, _, _, _, _ = deriv_total(soc_current, vrc_current, temp_current)

            s2  = soc_current  + 0.5 * h_sub * ds1
            v2  = vrc_current  + 0.5 * h_sub * dv1
            T2  = temp_current + 0.5 * h_sub * dT1
            # 对中间步使用硬约束（更稳定）
            T2  = np.clip(T2, -50.0, 120.0)
            ds2, dv2, dT2, _, _, _, _ = deriv_total(s2, v2, T2)

            s3  = soc_current  + 0.5 * h_sub * ds2
            v3  = vrc_current  + 0.5 * h_sub * dv2
            T3  = temp_current + 0.5 * h_sub * dT2
            T3  = np.clip(T3, -50.0, 120.0)
            ds3, dv3, dT3, _, _, _, _ = deriv_total(s3, v3, T3)

            s4  = soc_current  + h_sub * ds3
            v4  = vrc_current  + h_sub * dv3
            T4  = temp_current + h_sub * dT3
            T4  = np.clip(T4, -50.0, 120.0)
            ds4, dv4, dT4, IL_end, Voc_end, VL_end, Ploss_end = deriv_total(s4, v4, T4)

            soc_current  = soc_current  + (h_sub/6.0) * (ds1 + 2*ds2 + 2*ds3 + ds4)
            vrc_current  = vrc_current  + (h_sub/6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
            temp_current = temp_current + (h_sub/6.0) * (dT1 + 2*dT2 + 2*dT3 + dT4)
            
            # 每个子步后应用硬约束（用于中间计算）
            soc_current  = np.clip(soc_current, 0.05, 1.0)
            temp_current = np.clip(temp_current, -50.0, 120.0)

        return soc_current, vrc_current, temp_current, IL_end, Voc_end, VL_end, Ploss_end


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

        # ── Step 5: RK4求解耦合态 ──
        Q_other = (W_CPU    * P_cpu_e
                 + W_SCREEN * P_scr_e
                 + W_NET    * P_net_e
                 + W_GPS    * P_gps_e)

        soc_new, Vrc_new, T_new, IL_val, Voc_val, VL_val, Ploss_val = \
            rk4_step_ecm(soc, Vrc, T, P_load, Q_other, dt)
        
        # 对最终结果应用硬边界约束（用于确保数值稳定）
        soc_new = np.clip(soc_new, 0.05, 1.0)
        T_new = np.clip(T_new, -50.0, 120.0)

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
    # 使用 dt_ode=1.0 确保高精度的ODE数值积分
    res = run_joint_poisson_simulation(Q_aging=Q_ag, T_sim=86400, dt=10.0, dt_ode=1.0)
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
    100:   '#3498db',
    500:   '#f39c12',
    1000:  '#e67e22',
    5000.0: '#e74c3c',
}

AGING_LABELS = {
    0.0:    '新电池 (Q=0)',
    100.0: '轻度老化 (Q=100)',
    500.0:   '中度老化 (Q=500)',
    1000.0:  '重度老化 (Q=1000)',
    5000.0: '极度老化 (Q=5000)',
}


# ===============================================================
# 图 1: 联合泊松过程 - 各应用独立强度函数 λᵢ(t)
# ===============================================================
fig1, axes1 = plt.subplots(2, 1, figsize=(14, 10))
fig1.suptitle('联合泊松过程：各应用类型独立到达率函数', fontsize=15, fontweight='bold')

t_24h = np.linspace(0, 86400, 1000)
t_h_arr = t_24h / 3600.0

# (0) 各应用独立强度
for app_type in APP_TYPE_KEYS:
    lam_vals = [lambda_intensity_per_app(t, app_type) for t in t_24h]
    axes1[0].plot(t_h_arr, lam_vals, linewidth=2.0, 
                  color=APP_COLORS[app_type], label=APP_TYPES[app_type]['label'])

axes1[0].set_xlabel('时间 (h)', fontsize=11)
axes1[0].set_ylabel('到达强度 λi(t) (events/h)', fontsize=11)
axes1[0].set_title('各应用类型独立到达率', fontsize=12, fontweight='bold')
axes1[0].set_xticks(range(0, 25, 2))
axes1[0].legend(fontsize=9, loc='upper left', ncol=2)
axes1[0].grid(True, alpha=0.3)

# (1) 总到达率（所有应用的和）
lam_total_vals = [lambda_total(t) for t in t_24h]
axes1[1].fill_between(t_h_arr, lam_total_vals, alpha=0.3, color='#3498db')
axes1[1].plot(t_h_arr, lam_total_vals, linewidth=2.5, color='#2980b9')
axes1[1].set_xlabel('时间 (h)', fontsize=11)
axes1[1].set_ylabel('总到达强度 Σλi(t) (events/h)', fontsize=11)
axes1[1].set_title('联合泊松过程总到达率（所有应用的和）', fontsize=12, fontweight='bold')
axes1[1].set_xticks(range(0, 25, 2))
axes1[1].grid(True, alpha=0.3)
axes1[1].axvspan(0, 7, alpha=0.08, color='gray', label='深夜/凌晨')
axes1[1].axvspan(22, 24, alpha=0.08, color='gray')
axes1[1].legend(fontsize=10)

plt.tight_layout()
plt.show()


# ===============================================================
# 图 2: 多老化水平对比 — SOC / 温度 / 端电压 / 放电电流
# ===============================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.suptitle('不同老化水平下电池状态与温度演化对比（联合泊松过程）', fontsize=15, fontweight='bold')

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
axes2[0,0].set_xlabel('时间 (h)', fontsize=11)
axes2[0,0].set_ylabel('SOC', fontsize=11)
axes2[0,0].set_title('SOC 随时间变化', fontsize=12, fontweight='bold')
axes2[0,0].legend(fontsize=8, loc='best')
axes2[0,0].grid(True, alpha=0.3)
axes2[0,0].set_ylim(0, 1.05)

axes2[0,1].axhline(y=Tamb, color='gray', linestyle='--', linewidth=1, label=f'环境温度 {Tamb}°C')
axes2[0,1].set_xlabel('时间 (h)', fontsize=11)
axes2[0,1].set_ylabel('温度 T (°C)', fontsize=11)
axes2[0,1].set_title('电池温度随时间变化', fontsize=12, fontweight='bold')
axes2[0,1].legend(fontsize=8, loc='best')
axes2[0,1].grid(True, alpha=0.3)

axes2[1,0].set_xlabel('时间 (h)', fontsize=11)
axes2[1,0].set_ylabel('负载端电压 VL (V)', fontsize=11)
axes2[1,0].set_title('端电压随时间变化', fontsize=12, fontweight='bold')
axes2[1,0].legend(fontsize=8, loc='best')
axes2[1,0].grid(True, alpha=0.3)

axes2[1,1].set_xlabel('时间 (h)', fontsize=11)
axes2[1,1].set_ylabel('放电电流 IL (A)', fontsize=11)
axes2[1,1].set_title('放电电流随时间变化', fontsize=12, fontweight='bold')
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
fig3.suptitle('联合泊松过程：并行事件时间线与总功耗演化 (新电池)', fontsize=15, fontweight='bold')

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
# 图 4: 联合泊松过程特有 - 各应用到达次数分布
# ===============================================================
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle('联合泊松过程：各应用事件到达统计', fontsize=15, fontweight='bold')

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
axes4[0].set_ylabel('到达次数', fontsize=11)
axes4[0].set_title('各应用类型事件到达次数（24小时）', fontsize=12, fontweight='bold')
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
axes4[1].set_title('各应用活跃时间占比', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


# ===============================================================
# 图 5: 老化水平对比汇总柱状图
# ===============================================================
fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5.5))
fig5.suptitle('老化水平对比汇总（联合泊松过程）', fontsize=14, fontweight='bold')

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
# 最终汇总输出
# ===============================================================
print("\n" + "=" * 70)
print("联合泊松过程各老化水平仿真汇总")
print("=" * 70)
print(f"{'老化水平':<22} {'Qeff(Ah)':>9} {'放电时间(h)':>12} {'平均负载(mW)':>13} "
      f"{'最高温度(°C)':>13} {'最大并行':>8} {'总事件':>7}")
print("-" * 95)
for Q_ag in Q_AGING_LIST:
    r = all_results[Q_ag]
    print(f"{AGING_LABELS[Q_ag]:<22} "
          f"{calc_Qeff(Q_ag):>8.4f} "
          f"{r['discharge_time_h']:>11.2f} "
          f"{np.mean(r['P_total'])*1000:>12.1f} "
          f"{r['T'].max():>12.2f} "
          f"{int(r['n_active'].max()):>7} "
          f"{len(r['event_log']):>6}")
print("=" * 95)

print("\n" + "=" * 70)
print("新电池各应用到达统计:")
print("=" * 70)
r_new = all_results[0.0]
for app_type in APP_TYPE_KEYS:
    label = APP_TYPES[app_type]['label']
    count = r_new['arrival_counts'][app_type]
    active_time_h = r_new['app_active_time'][app_type] / 3600.0
    print(f"  {label:<12}: 到达{count:>4}次, 活跃时长{active_time_h:>6.2f}h")
print("=" * 70)