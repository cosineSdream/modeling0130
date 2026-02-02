import math
import numpy as np
import matplotlib.pyplot as plt


def trunc_norm(mu, sigma, lo, hi):
    if sigma <= 0:
        return float(np.clip(mu, lo, hi))
    x = mu + sigma * np.random.randn()
    return float(np.clip(x, lo, hi))


def Qeff_from_age(par):
    Qeff = par['Q0_Ah'] * (1 - par['k_age'] * math.sqrt(max(par['Qage_Ah'], 0) / (2 * par['Q0_Ah'])))
    return max(Qeff, 0.1)


def ecm_params_from_soc(S, par):
    S = float(np.clip(S, 0.05, 1.0))
    R0 = np.polyval(par['R0_poly'], S)
    R1 = np.polyval(par['R1_poly'], S)
    C1 = np.polyval(par['C1_poly'], S)
    R0 = max(R0, 1e-6)
    R1 = max(R1, 1e-6)
    C1 = max(C1, 1e-6)
    return R0, R1, C1


def ocv_params_from_T(T_C, par):
    Qage = par['Qage_Ah']

    p1 = (par['p1_c0'] + par['p1_c1'] * T_C + par['p1_c2'] * Qage +
          par['p1_c3'] * T_C ** 2 + par['p1_c4'] * T_C * Qage + par['p1_c5'] * Qage ** 2)

    p2 = (par['p2_c0'] + par['p2_c1'] * T_C + par['p2_c2'] * T_C * Qage +
          par['p2_c3'] * T_C ** 3 + par['p2_c4'] * T_C ** 2 * Qage + par['p2_c5'] * T_C * Qage ** 2)

    lam1_base = par['lam1_a1'] * T_C ** 2 + par['lam1_a2'] * T_C + par['lam1_a3']
    lam2_base = (par['lam2_b1'] * Qage ** 3 + par['lam2_b2'] * Qage ** 2 +
                 par['lam2_b3'] * Qage + par['lam2_b4'])

    scale = par['Qref_Ah'] / par['Q0_Ah']
    lam1 = lam1_base * scale
    lam2 = lam2_base * scale
    return p1, p2, lam1, lam2


def Voc_from_soc(S, T_C, par):
    Qeff = Qeff_from_age(par)
    p1, p2, lam1, lam2 = ocv_params_from_T(T_C, par)
    q = (1 - S) * Qeff
    # guard against overflow in exp
    val = 0.0
    try:
        val = p1 * (math.exp(lam1 * q) - 1) + p2 * (math.exp(lam2 * q) - 1) + par['Voc0']
    except OverflowError:
        # if overflow, use large-value approximation
        val = par['Voc0'] + np.sign(lam1) * np.inf
    return float(val)


def P_screen(B, par):
    return par['Pscreen_max_W'] * B


def P_cpu(u, par):
    return par['b_cpu_W'] + (par['Pcpu_max_W'] - par['b_cpu_W']) * u


def P_gps(state, par):
    if str(state).lower() == 'locating':
        return par['Pgps_locating_W']
    return par['Pgps_off_W']


def P_net(net_state, r, par):
    s = str(net_state).upper()
    if s == 'WIFI6':
        a = par['net']['WIFI6']['a_W']; b = par['net']['WIFI6']['b_W']
    elif s in ('4G_LTE_A', 'LTE4G'):
        a = par['net']['LTE4G']['a_W']; b = par['net']['LTE4G']['b_W']
    elif s in ('5GNSA', '5G_NSA', 'NSA5G'):
        a = par['net']['NSA5G']['a_W']; b = par['net']['NSA5G']['b_W']
    else:
        a = par['net']['WIFI6']['a_W']; b = par['net']['WIFI6']['b_W']
    return a * r + b


def deriv(x, Pload_W, Pscr, Pcpu, Pnet, Pgps, par):
    S = x[0]; Vrc = x[1]; T_C = x[2]
    Qeff = Qeff_from_age(par)
    R0, R1, C1 = ecm_params_from_soc(S, par)
    gamma = R1 * C1
    Voc = Voc_from_soc(S, T_C, par)
    Vt = Voc - Vrc
    disc = Vt * Vt - 4 * R0 * Pload_W
    if disc < 0:
        disc = 0.0
    IL = (Vt - math.sqrt(disc)) / (2 * R0)
    dsdt = -IL / Qeff / 3600.0
    dVrcdt = (IL * R1 - Vrc) / gamma
    VL = Vt - IL * R0
    Ploss = (Voc - VL) * IL
    hA = par['h_Wm2K'] * par['A_m2']
    Pheat = Ploss + par['eta_scr'] * Pscr + par['eta_cpu'] * Pcpu + par['eta_net'] * Pnet + par['eta_gps'] * Pgps
    dTdt = (Pheat - hA * (T_C - par['Tamb_C'])) / par['Cth_JK']
    dx = np.array([dsdt, dVrcdt, dTdt], dtype=float)
    return dx, dsdt, dVrcdt, dTdt, IL, Voc, Vt


def simulate_one_mode(B, u, r, net_state, gps_state, par):
    Pbase_scr = P_screen(0.0, par)
    Pbase_cpu = P_cpu(par['u_idle'], par)
    Pbase_net = P_net('WIFI6', 0.0, par)
    Pbase_gps = P_gps('off', par)
    P_base_W = Pbase_scr + Pbase_cpu + Pbase_net + Pbase_gps
    Pscr = P_screen(B, par)
    Pcpu = P_cpu(u, par)
    Pnet = P_net(net_state, r, par)
    Pgps = P_gps(gps_state, par)
    P_inc = (Pscr - Pbase_scr) + (Pcpu - Pbase_cpu) + (Pnet - Pbase_net) + (Pgps - Pbase_gps)
    Pload_W = max(P_base_W + P_inc, 0.0)

    S = par['S0']
    Vrc = 0.0
    T_C = par['T0_C']
    dt = par['dt']
    nmax = int(math.ceil(par['tmax_h'] * 3600.0 / dt))

    t_arr = []
    S_arr = []
    Vt_arr = []
    Voc_arr = []
    IL_arr = []
    Vrc_arr = []
    T_arr = []
    P_arr = []

    for step in range(1, nmax + 1):
        if S <= par['S_min']:
            break
        t = (step - 1) * dt
        x = np.array([S, Vrc, T_C], dtype=float)
        f = lambda xx: deriv(xx, Pload_W, Pscr, Pcpu, Pnet, Pgps, par)[0]
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        S = float(np.clip(x_next[0], 0.0, 1.0))
        Vrc = float(x_next[1])
        T_C = float(x_next[2])
        _, _, _, _, IL, Voc, Vt = deriv(x, Pload_W, Pscr, Pcpu, Pnet, Pgps, par)
        t_arr.append(t)
        S_arr.append(S)
        Vt_arr.append(Vt)
        Voc_arr.append(Voc)
        IL_arr.append(IL)
        Vrc_arr.append(Vrc)
        T_arr.append(T_C)
        P_arr.append(Pload_W)

    sim = {
        't_s': np.array(t_arr),
        't_h': np.array(t_arr) / 3600.0,
        'S': np.array(S_arr),
        'Vt': np.array(Vt_arr),
        'Voc': np.array(Voc_arr),
        'IL': np.array(IL_arr),
        'Vrc': np.array(Vrc_arr),
        'T_C': np.array(T_arr),
        'Ptotal_W': np.array(P_arr),
        'B': B,
        'u': u,
        'r': r,
        'Pscr_W': Pscr,
        'Pcpu_W': Pcpu,
        'Pnet_W': Pnet,
        'Pgps_W': Pgps,
        'Pbase_W': P_base_W,
        'Pload_W': Pload_W,
    }
    return sim


# ---------- Joint Poisson process: multi-activity scene simulation ----------
APP_TYPES = {
    'video':    {'label': 'Video',    'B_mean': 0.72, 'B_std': 0.08, 'u_mean': 0.22, 'u_std': 0.04, 'r_mean': 0.70, 'r_std': 0.10, 'net_state': 'WIFI6', 'gps_state': 'off', 'dur_mean': 4800.0, 'dur_std': 2400.0, 'lambda_weight': 0.22},
    'gaming':   {'label': 'Gaming',   'B_mean': 0.85, 'B_std': 0.05, 'u_mean': 0.65, 'u_std': 0.08, 'r_mean': 0.45, 'r_std': 0.12, 'net_state': 'WIFI6', 'gps_state': 'off', 'dur_mean': 3600.0, 'dur_std': 1800.0, 'lambda_weight': 0.15},
    'social':   {'label': 'Social',   'B_mean': 0.60, 'B_std': 0.10, 'u_mean': 0.18, 'u_std': 0.05, 'r_mean': 0.35, 'r_std': 0.08, 'net_state': '4G_LTE_A', 'gps_state': 'off', 'dur_mean': 2400.0, 'dur_std': 1200.0, 'lambda_weight': 0.28},
    'navigation':{'label': 'Navigation','B_mean':0.75,'B_std':0.06,'u_mean':0.35,'u_std':0.06,'r_mean':0.25,'r_std':0.05,'net_state':'4G_LTE_A','gps_state':'locating','dur_mean':1800.0,'dur_std':900.0,'lambda_weight':0.10},
    'calling':  {'label': 'Calling',  'B_mean': 0.15, 'B_std': 0.05, 'u_mean': 0.12, 'u_std': 0.03, 'r_mean': 0.08, 'r_std': 0.02, 'net_state': '4G_LTE_A', 'gps_state': 'off', 'dur_mean': 600.0, 'dur_std': 300.0, 'lambda_weight': 0.12},
    'music':    {'label': 'Music',    'B_mean': 0.20, 'B_std': 0.08, 'u_mean': 0.08, 'u_std': 0.02, 'r_mean': 0.15, 'r_std': 0.04, 'net_state': 'WIFI6', 'gps_state': 'off', 'dur_mean': 7200.0, 'dur_std': 3600.0, 'lambda_weight': 0.08},
    'bg_download':{'label':'BG download','B_mean':0.0,'B_std':0.0,'u_mean':0.15,'u_std':0.03,'r_mean':0.85,'r_std':0.08,'net_state':'WIFI6','gps_state':'off','dur_mean':1200.0,'dur_std':600.0,'lambda_weight':0.03},
    'bg_sync':  {'label':'BG sync',  'B_mean':0.0,'B_std':0.0,'u_mean':0.05,'u_std':0.01,'r_mean':0.20,'r_std':0.05,'net_state':'WIFI6','gps_state':'off','dur_mean':180.0,'dur_std':90.0,'lambda_weight':0.02},
}

APP_KEYS = list(APP_TYPES.keys())
# normalize weights
tot_w = sum(APP_TYPES[k]['lambda_weight'] for k in APP_KEYS)
for k in APP_KEYS:
    APP_TYPES[k]['lambda_weight'] /= tot_w


def lambda_intensity_per_app(t_sec, app_key):
    """Return intensity (events/hour) for given app at time t_sec."""
    app = APP_TYPES[app_key]
    t_h = (t_sec % 86400) / 3600.0
    peak1 = 8.0 * math.exp(-0.5 * ((t_h - 13.0) / 3.5) ** 2)
    peak2 = 10.0 * math.exp(-0.5 * ((t_h - 21.0) / 2.5) ** 2)
    base_intensity = 0.5 + peak1 + peak2
    # simple time modulation per app type (copied idea)
    if app_key == 'video':
        time_mod = 1.0 + 0.5 * math.exp(-0.5 * ((t_h - 20.0) / 3.0) ** 2)
    elif app_key == 'gaming':
        time_mod = 1.0 + 0.3 * (math.exp(-0.5 * ((t_h - 15.0) / 2.5) ** 2) + math.exp(-0.5 * ((t_h - 22.0) / 2.0) ** 2))
    elif app_key == 'social':
        time_mod = 1.0 + 0.2 * (math.exp(-0.5 * ((t_h - 8.0) / 2.0) ** 2) + math.exp(-0.5 * ((t_h - 19.0) / 2.5) ** 2))
    elif app_key == 'navigation':
        time_mod = 1.0 + 0.8 * (math.exp(-0.5 * ((t_h - 8.0) / 1.5) ** 2) + math.exp(-0.5 * ((t_h - 18.0) / 1.5) ** 2))
    elif app_key == 'calling':
        morning = 1.0 / (1.0 + math.exp(-3.0 * (t_h - 9.0)))
        evening = 1.0 / (1.0 + math.exp(3.0 * (t_h - 18.0)))
        time_mod = 0.8 + 0.5 * morning * evening
    elif app_key == 'music':
        time_mod = 1.0 + 0.4 * (math.exp(-0.5 * ((t_h - 8.5) / 2.0) ** 2) + math.exp(-0.5 * ((t_h - 14.0) / 3.0) ** 2))
    elif app_key in ['bg_download', 'bg_sync']:
        if 1 <= t_h <= 6:
            time_mod = 1.2
        else:
            time_mod = 0.9
    else:
        time_mod = 1.0
    return base_intensity * app['lambda_weight'] * time_mod


def sample_event(t_start, app_key):
    app = APP_TYPES[app_key]
    mean_d = app['dur_mean']; std_d = app['dur_std']
    sigma2 = math.log(1 + (std_d / mean_d) ** 2) if mean_d > 0 else 0.0
    mu = math.log(mean_d) - sigma2 / 2 if mean_d > 0 else 0.0
    D_k = max(float(np.random.lognormal(mean=mu, sigma=math.sqrt(sigma2))) if sigma2 > 0 else mean_d, 10.0)
    B_k = float(np.clip(np.random.normal(app['B_mean'], app['B_std']), 0.0, 1.0))
    u_k = float(np.clip(np.random.normal(app['u_mean'], app['u_std']), 0.00, 1.0))
    r_k = float(np.clip(np.random.normal(app['r_mean'], app['r_std']), 0.0, 1.0))
    return {'id': id(object()), 'app_type': app_key, 'label': app['label'], 't_start': t_start, 'duration': D_k, 't_end': t_start + D_k, 'B': B_k, 'u': u_k, 'r': r_k, 'net_state': app['net_state'], 'gps_state': app['gps_state']}


def calc_module_powers(active_set, par):
    Pscr_sum = 0.0; Pcpu_sum = 0.0; Pnet_sum = 0.0; Pgps_sum = 0.0
    for ev in active_set:
        Pscr_sum += P_screen(ev['B'], par)
        Pcpu_sum += P_cpu(ev['u'], par) - P_cpu(par['u_idle'], par)
        Pnet_sum += P_net(ev['net_state'], ev['r'], par) - P_net('WIFI6', 0.0, par)
        Pgps_sum += P_gps(ev['gps_state'], par) - P_gps('off', par)
    Pcpu_sum += P_cpu(par['u_idle'], par)
    Pscr_sum += P_screen(0.0, par)
    Pnet_sum += P_net('WIFI6', 0.0, par)
    Pgps_sum += P_gps('off', par)
    return Pcpu_sum, Pscr_sum, Pnet_sum, Pgps_sum


def simulate_one_scene(par, T_sim_h=12.0, dt=1.0):
    """Simulate a scene composed of random arrival sequences from multiple activities.

    Returns a dict with time series similar to other simulators.
    """
    T_sim = float(T_sim_h) * 3600.0
    n_steps = int(math.ceil(T_sim / dt))
    x = np.array([par['S0'], 0.0, par.get('T0_C', par.get('Tamb_C', 25.0))], dtype=float)
    t_list = []
    S_list = []
    Vt_list = []
    Voc_list = []
    IL_list = []
    Vrc_list = []
    T_list = []
    P_list = []
    active_counts = []

    active_events = []

    # precompute base components
    Pbase_scr = P_screen(0.0, par)
    Pbase_cpu = P_cpu(par['u_idle'], par)
    Pbase_net = P_net('WIFI6', 0.0, par)
    Pbase_gps = P_gps('off', par)
    P_base_W = Pbase_scr + Pbase_cpu + Pbase_net + Pbase_gps

    for step in range(n_steps):
        t = step * dt
        # arrivals: Poisson with rate (events/hour) -> mean per dt = lam * dt/3600
        for appk in APP_KEYS:
            lam_h = lambda_intensity_per_app(t, appk)
            mean_arrivals = lam_h * (dt / 3600.0)
            if mean_arrivals <= 0:
                continue
            n_arr = np.random.poisson(mean_arrivals)
            for _ in range(n_arr):
                ev = sample_event(t, appk)
                active_events.append(ev)

        # remove finished events
        active_events = [ev for ev in active_events if ev['t_end'] > t]

        # compute module powers
        Pcpu_m, Pscr_m, Pnet_m, Pgps_m = calc_module_powers(active_events, par)
        Pload_W = max(Pcpu_m + Pscr_m + Pnet_m + Pgps_m, 0.0)

        # advance thermal/electric state with RK4 using deriv
        f = lambda xx: deriv(xx, Pload_W, Pscr_m, Pcpu_m, Pnet_m, Pgps_m, par)[0]
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        S = float(np.clip(x[0], 0.0, 1.0))
        Vrc = float(x[1])
        T_C = float(x[2])
        _, _, _, _, IL, Voc, Vt = deriv(x, Pload_W, Pscr_m, Pcpu_m, Pnet_m, Pgps_m, par)

        t_list.append(t)
        S_list.append(S)
        Vt_list.append(Vt)
        Voc_list.append(Voc)
        IL_list.append(IL)
        Vrc_list.append(Vrc)
        T_list.append(T_C)
        P_list.append(Pload_W)
        active_counts.append(len(active_events))

        if S <= par.get('S_min', 0.05):
            break

    sim = {
        't_s': np.array(t_list),
        't_h': np.array(t_list) / 3600.0,
        'S': np.array(S_list),
        'Vt': np.array(Vt_list),
        'Voc': np.array(Voc_list),
        'IL': np.array(IL_list),
        'Vrc': np.array(Vrc_list),
        'T_C': np.array(T_list),
        'Ptotal_W': np.array(P_list),
        'active_counts': np.array(active_counts),
        'events': active_events,
    }
    return sim


def main():
    np.random.seed(7)
    par = {}
    par['Q0_Ah'] = 5.0
    par['Qage_Ah'] = 0.0
    par['k_age'] = 0.004156
    par['S0'] = 1.0
    par['S_min'] = 0.05
    par['dt'] = 1.0
    par['tmax_h'] = 12.0
    par['Tamb_C'] = 25.0
    par['T0_C'] = par['Tamb_C']
    par['p1_c0'] = 0.5385
    par['p1_c1'] = 6.234e-4
    par['p1_c2'] = -2.608e-7
    par['p1_c3'] = 2.191e-5
    par['p1_c4'] = 8.627e-8
    par['p1_c5'] = 8.928e-11
    par['p2_c0'] = -3.189e-11
    par['p2_c1'] = 9.634e-13
    par['p2_c2'] = -5.539e-16
    par['p2_c3'] = -3.508e-16
    par['p2_c4'] = 2.347e-17
    par['p2_c5'] = 1.431e-20
    par['lam1_a1'] = -6.573e-5
    par['lam1_a2'] = 5.767e-3
    par['lam1_a3'] = -0.3226
    par['lam2_b1'] = 1.299e-13
    par['lam2_b2'] = -2.652e-9
    par['lam2_b3'] = 3.129e-5
    par['lam2_b4'] = 2.330
    par['Voc0'] = 4.2
    par['Qref_Ah'] = 10.0
    par['R0_poly'] = np.array([-2.2474747475e-02, 4.1266233766e-02, -3.2458152958e-02, 3.7307142857e-02])
    par['R1_poly'] = np.array([-3.1717171717e-01, 5.9903679654e-01, -3.6628174603e-01, 1.0417619048e-01])
    par['C1_poly'] = np.array([5.0124579125e+02, -1.0881944444e+03, 1.0570180976e+03, 3.3737777778e+02])
    par['Cth_JK'] = 90.0
    par['h_Wm2K'] = 20.0
    par['A_m2'] = 0.0128
    par['eta_cpu'] = 1.0
    par['eta_scr'] = 0.2
    par['eta_net'] = 0.7
    par['eta_gps'] = 0.7
    par['Pscreen_max_W'] = 2.464
    par['b_cpu_W'] = 0.1
    par['Pcpu_max_W'] = 4.0
    par['Pgps_locating_W'] = 0.25
    par['Pgps_off_W'] = 0.0
    par['net'] = {
        'WIFI6': {'a_W': (300 - 11.5) / 1000.0, 'b_W': 11.5 / 1000.0},
        'LTE4G': {'a_W': (400 - 30) / 1000.0, 'b_W': 30 / 1000.0},
        'NSA5G': {'a_W': 510 / 1000.0, 'b_W': 40 / 1000.0},
    }
    par['u_idle'] = 0.00

    modes = [
        {'key': 'video', 'label': 'Video playback', 'B_mean': 0.72, 'B_std': 0.07, 'u_mean': 0.22, 'u_std': 0.02, 'r_mean': 0.70, 'r_std': 0.07, 'net': 'WIFI6', 'gps': 'off'},
        {'key': 'gaming', 'label': 'Gaming', 'B_mean': 0.85, 'B_std': 0.09, 'u_mean': 0.45, 'u_std': 0.05, 'r_mean': 0.50, 'r_std': 0.05, 'net': 'WIFI6', 'gps': 'off'},
        {'key': 'social', 'label': 'Social browsing', 'B_mean': 0.60, 'B_std': 0.06, 'u_mean': 0.15, 'u_std': 0.02, 'r_mean': 0.45, 'r_std': 0.05, 'net': 'WIFI6', 'gps': 'off'},
        {'key': 'navigation', 'label': 'Navigation', 'B_mean': 0.60, 'B_std': 0.06, 'u_mean': 0.20, 'u_std': 0.02, 'r_mean': 0.10, 'r_std': 0.01, 'net': '4G_LTE_A', 'gps': 'locating'},
        {'key': 'calling', 'label': 'Voice call', 'B_mean': 0.40, 'B_std': 0.04, 'u_mean': 0.15, 'u_std': 0.02, 'r_mean': 0.20, 'r_std': 0.02, 'net': '4G_LTE_A', 'gps': 'off'},
        {'key': 'music', 'label': 'Music (screen off)', 'B_mean': 0.00, 'B_std': 0.00, 'u_mean': 0.20, 'u_std': 0.02, 'r_mean': 0.30, 'r_std': 0.03, 'net': 'WIFI6', 'gps': 'off'},
        {'key': 'screenoff', 'label': 'Screen off (idle)', 'B_mean': 0.00, 'B_std': 0.00, 'u_mean': 0.05, 'u_std': 0.01, 'r_mean': 0.05, 'r_std': 0.01, 'net': 'WIFI6', 'gps': 'off'},
    ]

    soc_grid = np.linspace(0.05, 1.0, 400)
    T_grid_C = par['Tamb_C'] * np.ones_like(soc_grid)
    Voc_grid = np.array([Voc_from_soc(s, T_grid_C[0], par) for s in soc_grid])
    plt.figure('Voc vs SOC')
    plt.plot(soc_grid, Voc_grid, linewidth=2)
    plt.grid(True)
    plt.xlabel('SOC S')
    plt.ylabel('Voc (V)')
    plt.title('OCV curve: Voc(S)')

    plt.figure('SOC vs time for each mode')
    plt.clf();
    results = {}
    for md in modes:
        B = trunc_norm(md['B_mean'], md['B_std'], 0.0, 1.0)
        if md['key'] == 'screenoff':
            u_lo = 0.0
        else:
            u_lo = 0.02
        u = trunc_norm(md['u_mean'], md['u_std'], u_lo, 1.0)
        r = trunc_norm(md['r_mean'], md['r_std'], 0.0, 1.0)
        sim = simulate_one_mode(B, u, r, md['net'], md['gps'], par)
        results[md['key']] = sim
        plt.plot(sim['t_h'], sim['S'], linewidth=2, label=md['label'])

    plt.xlabel('time (h)')
    plt.ylabel('SOC S')
    plt.title('SOC decay under different usage modes')
    plt.legend(loc='best')

    plt.figure('Temperature vs time for each mode')
    plt.clf()
    for md in modes:
        sim = results[md['key']]
        plt.plot(sim['t_h'], sim['T_C'], linewidth=2, label=md['label'])
    plt.xlabel('time (h)')
    plt.ylabel('Cell Temperature T (°C)')
    plt.title('Temperature rise under different usage modes')
    plt.legend(loc='best')

    print('--- Mode summary (sampled B,u,r and mean power) ---')
    for md in modes:
        sim = results[md['key']]
        meanP_mW = np.mean(sim['Ptotal_W']) * 1000.0 if sim['Ptotal_W'].size > 0 else 0.0
        discharge_time = sim['t_h'][-1] if sim['t_h'].size > 0 else 0.0
        print(f"{md['label']}: B={sim['B']:.3f}, u={sim['u']:.3f}, r={sim['r']:.3f}, mean P={meanP_mW:.1f} mW, discharge time={discharge_time:.2f} h")

    plt.figure('Temperature vs time (first 30 min)')
    plt.clf()
    t_end_h = 0.5
    for md in modes:
        sim = results[md['key']]
        idx = sim['t_h'] <= t_end_h
        t_min = sim['t_h'][idx] * 60.0
        T_C = sim['T_C'][idx]
        plt.plot(t_min, T_C, linewidth=2, label=md['label'])
    plt.xlabel('time (min)')
    plt.ylabel('Cell Temperature T (°C)')
    plt.title('Temperature rise under different usage modes (first 30 min)')
    plt.legend(loc='best')
    # --- run joint Poisson scene simulation and visualize ---
    sim_scene = simulate_one_scene(par, T_sim_h=12.0, dt=1.0)

    plt.figure('Scene: SOC vs time')
    plt.clf()
    plt.plot(sim_scene['t_h'], sim_scene['S'], linewidth=2, label='Joint Poisson Scene')
    plt.xlabel('time (h)')
    plt.ylabel('SOC S')
    plt.title('SOC decay under joint Poisson scene')
    plt.legend(loc='best')

    plt.figure('Scene: Temperature vs time')
    plt.clf()
    plt.plot(sim_scene['t_h'], sim_scene['T_C'], linewidth=2, label='Joint Poisson Scene')
    plt.xlabel('time (h)')
    plt.ylabel('Cell Temperature T (°C)')
    plt.title('Temperature under joint Poisson scene')
    plt.legend(loc='best')

    plt.figure('Scene: Active events and power')
    plt.clf()
    ax1 = plt.gca()
    ax1.plot(sim_scene['t_h'], sim_scene['active_counts'], color='C0', label='Active events')
    ax1.set_xlabel('time (h)')
    ax1.set_ylabel('Active events', color='C0')
    ax2 = ax1.twinx()
    ax2.plot(sim_scene['t_h'], sim_scene['Ptotal_W'] * 1000.0, color='C1', label='Total power (mW)')
    ax2.set_ylabel('Power (mW)', color='C1')
    plt.title('Active events and total power')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    main()
