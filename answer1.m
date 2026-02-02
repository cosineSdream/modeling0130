clear; clc; close all;
rng(7); % 固定随机种子，便于复现实验

%% -------------------------
% 0) Parameters (tune here)
%% -------------------------
par = struct();

% ---- Battery capacity / aging ----
par.Q0_Ah = 5.0;
par.Qage_Ah = 0.0;
par.k_age = 0.004156;

% ---- SOC integration ----
par.S0 = 1.0;
par.S_min = 0.05;
par.dt = 1.0;
par.tmax_h = 12.0;

% ---- Ambient temperature ----
par.Tamb_C = 25.0;
par.T0_C = par.Tamb_C;

% ---- OCV parameter coefficients (T is REAL-TIME cell temp, Qage fixed in one run) ----
% p1(T, Qage) = c10 + c11*T + c12*Qage + c13*T^2 + c14*T*Qage + c15*Qage^2
par.p1_c0 = 0.5385;
par.p1_c1 = 6.234e-4;
par.p1_c2 = -2.608e-7;
par.p1_c3 = 2.191e-5;
par.p1_c4 = 8.627e-8;
par.p1_c5 = 8.928e-11;

% p2(T, Qage) = c20 + c21*T + c22*T*Qage + c23*T^3 + c24*T^2*Qage + c25*T*Qage^2
par.p2_c0 = -3.189e-11;
par.p2_c1 = 9.634e-13;
par.p2_c2 = -5.539e-16;
par.p2_c3 = -3.508e-16;
par.p2_c4 = 2.347e-17;
par.p2_c5 = 1.431e-20;

% lambda1_base(T) = a1*T^2 + a2*T + a3
par.lam1_a1 = -6.573e-5;
par.lam1_a2 = 5.767e-3;
par.lam1_a3 = -0.3226;

% lambda2_base(Qage) = b1*Qage^3 + b2*Qage^2 + b3*Qage + b4
par.lam2_b1 = 1.299e-13;
par.lam2_b2 = -2.652e-9;
par.lam2_b3 = 3.129e-5;
par.lam2_b4 = 2.330;

% OCV offset (keep as tunable parameter)
par.Voc0 = 3.65;

% scaling: lambda = lambda_base * (Qref/Q0)
par.Qref_Ah = 10.0;

par.Voc0 = 4.2; % 你当前代码这样写，我保持不动
par.Qref_Ah = 10.0; % 你当前代码这样写，我保持不动

% ---- ECM parameters: cubic polynomials in SOC S ----
par.R0_poly = [-2.2474747475e-02, 4.1266233766e-02, -3.2458152958e-02, 3.7307142857e-02];
par.R1_poly = [-3.1717171717e-01, 5.9903679654e-01, -3.6628174603e-01, 1.0417619048e-01];
par.C1_poly = [ 5.0124579125e+02, -1.0881944444e+03, 1.0570180976e+03, 3.3737777778e+02];

% ---- Thermal model ----
par.Cth_JK = 90.0;
par.h_Wm2K = 20.0;
par.A_m2 = 0.0128; % 你目前代码取值保持不动
par.eta_cpu = 1.0;
par.eta_scr = 0.2;
par.eta_net = 0.7;
par.eta_gps = 0.7;

% ---- Power models (you can tune) ----
% Screen: P_screen = Pscreen_max * B
par.Pscreen_max_W = 2.464;

% CPU: linear interpolation
par.b_cpu_W = 0.1; % b_cpu
par.Pcpu_max_W = 4.0; % Pcpu,max

% GPS: locating constant power
par.Pgps_locating_W = 0.25;
par.Pgps_off_W = 0.0;

% Network: Pinternet = a*r + b
par.net = struct();
par.net.WIFI6 = struct('a_W', (300-11.5)/1000, 'b_W', 11.5/1000);
par.net.LTE4G = struct('a_W', (400-30)/1000, 'b_W', 30/1000);
par.net.NSA5G = struct('a_W', 510/1000, 'b_W', 40/1000);

% ---- Base standby ----
par.u_idle = 0.00;%无用参数，直接设为0

%% -------------------------
% 1) Mode definitions
%% -------------------------
modes = { ...
struct('key','video', 'label','Video playback', 'B_mean',0.72,'B_std',0.07,'u_mean',0.22,'u_std',0.02,'r_mean',0.70,'r_std',0.07,'net','WIFI6','gps','off'); ...
struct('key','gaming', 'label','Gaming', 'B_mean',0.85,'B_std',0.09,'u_mean',0.45,'u_std',0.05,'r_mean',0.50,'r_std',0.05,'net','WIFI6','gps','off'); ...
struct('key','social', 'label','Social browsing', 'B_mean',0.60,'B_std',0.06,'u_mean',0.15,'u_std',0.02,'r_mean',0.45,'r_std',0.05,'net','WIFI6','gps','off'); ...
struct('key','navigation', 'label','Navigation', 'B_mean',0.60,'B_std',0.06,'u_mean',0.20,'u_std',0.02,'r_mean',0.10,'r_std',0.01,'net','4G_LTE_A','gps','locating'); ...
struct('key','calling', 'label','Voice call', 'B_mean',0.40,'B_std',0.04,'u_mean',0.15,'u_std',0.02,'r_mean',0.20,'r_std',0.02,'net','4G_LTE_A','gps','off'); ...
struct('key','music', 'label','Music (screen off)', 'B_mean',0.00,'B_std',0.00,'u_mean',0.20,'u_std',0.02,'r_mean',0.30,'r_std',0.03,'net','WIFI6','gps','off'); ...
% >>> NEW: screen-off idle mode (all module means/stds are 0) <<<
struct('key','screenoff', 'label','Screen off (idle)', 'B_mean',0.00,'B_std',0.00,'u_mean',0.05,'u_std',0.01,'r_mean',0.05,'r_std',0.01,'net','WIFI6','gps','off') ...
};

%% -------------------------
% 2) Plot Voc vs SOC
%% -------------------------
soc_grid = linspace(0.05, 1.0, 400);
T_grid_C = par.Tamb_C * ones(size(soc_grid));
Voc_grid = arrayfun(@(s) Voc_from_soc(s, T_grid_C(1), par), soc_grid);

figure('Name','Voc vs SOC');
plot(soc_grid, Voc_grid, 'LineWidth', 2);
grid on;
xlabel('SOC S'); ylabel('Voc (V)');
title('OCV curve: Voc(S)');

%% -------------------------
% 3) Simulate each mode and plot SOC(t)
%% -------------------------
figure('Name','SOC vs time for each mode'); hold on; grid on;

results = struct();
for i = 1:numel(modes)
md = modes{i};

% sample (B,u,r) once per mode, truncated
B = trunc_norm(md.B_mean, md.B_std, 0.0, 1.0);

% >>> MODIFIED: for screenoff mode, allow u to be 0 <<<
if strcmpi(md.key,'screenoff')
u_lo = 0.0;
else
u_lo = 0.02;
end
u = trunc_norm(md.u_mean, md.u_std, u_lo, 1.0);

r = trunc_norm(md.r_mean, md.r_std, 0.0, 1.0);

% run simulation
sim = simulate_one_mode(B, u, r, md.net, md.gps, par);

results.(md.key) = sim;
plot(sim.t_h, sim.S, 'LineWidth', 2, 'DisplayName', md.label);
end

xlabel('time (h)'); ylabel('SOC S');
title('SOC decay under different usage modes');
legend('Location','best');

%% -------------------------
% 4) Plot Temperature vs time for each mode
%% -------------------------
figure('Name','Temperature vs time for each mode'); hold on; grid on;

for i = 1:numel(modes)
md = modes{i};
sim = results.(md.key);
plot(sim.t_h, sim.T_C, 'LineWidth', 2, 'DisplayName', md.label);
end

xlabel('time (h)'); ylabel('Cell Temperature T (^{\circ}C)');
title('Temperature rise under different usage modes');
legend('Location','best');

% 打印每个模式的采样参数和平均功率
disp('--- Mode summary (sampled B,u,r and mean power) ---');
for i = 1:numel(modes)
md = modes{i};
sim = results.(md.key);
fprintf('%s: B=%.3f, u=%.3f, r=%.3f, mean P=%.1f mW, discharge time=%.2f h\n', ...
md.label, sim.B, sim.u, sim.r, mean(sim.Ptotal_W)*1000, sim.t_h(end));
end

figure('Name','Temperature vs time (first 30 min)'); hold on; grid on;

t_end_h = 0.5; % 30 min = 0.5 h

for i = 1:numel(modes)
md = modes{i};
sim = results.(md.key);

idx = sim.t_h <= t_end_h;
t_min = sim.t_h(idx) * 60;
T_C = sim.T_C(idx);

plot(t_min, T_C, 'LineWidth', 2, 'DisplayName', md.label);
end

xlabel('time (min)'); ylabel('Cell Temperature T (^{\circ}C)');
title('Temperature rise under different usage modes (first 30 min)');
legend('Location','best');

%% ============================================================
% Local functions
%% ============================================================
function x = trunc_norm(mu, sigma, lo, hi)
if sigma <= 0
x = min(max(mu, lo), hi);
return;
end
x = mu + sigma*randn();
x = min(max(x, lo), hi);
end

function Qeff = Qeff_from_age(par)
Qeff = par.Q0_Ah * (1 - par.k_age * sqrt( max(par.Qage_Ah,0) / (2*par.Q0_Ah) ));
Qeff = max(Qeff, 0.1);
end

function [R0, R1, C1] = ecm_params_from_soc(S, par)
S = min(max(S, 0.05), 1.0);
R0 = polyval(par.R0_poly, S);
R1 = polyval(par.R1_poly, S);
C1 = polyval(par.C1_poly, S);
R0 = max(R0, 1e-6);
R1 = max(R1, 1e-6);
C1 = max(C1, 1e-6);
end

function Voc = Voc_from_soc(S, T_C, par)
Qeff = Qeff_from_age(par);
[p1, p2, lambda1, lambda2] = ocv_params_from_T(T_C, par);
q = (1 - S) * Qeff;
Voc = p1*(exp(lambda1*q)-1) + p2*(exp(lambda2*q)-1) + par.Voc0;
end

function P = P_screen(B, par)
P = par.Pscreen_max_W * B;
end

function P = P_cpu(u, par)
P = par.b_cpu_W + (par.Pcpu_max_W - par.b_cpu_W) .* u;
end

function P = P_gps(state, par)
if strcmpi(state,'locating')
P = par.Pgps_locating_W;
else
P = par.Pgps_off_W;
end
end

function P = P_net(net_state, r, par)
switch upper(net_state)
case 'WIFI6'
a = par.net.WIFI6.a_W; b = par.net.WIFI6.b_W;
case {'4G_LTE_A','LTE4G'}
a = par.net.LTE4G.a_W; b = par.net.LTE4G.b_W;
case {'5GNSA','5G_NSA','NSA5G'}
a = par.net.NSA5G.a_W; b = par.net.NSA5G.b_W;
otherwise
a = par.net.WIFI6.a_W; b = par.net.WIFI6.b_W;
end
P = a*r + b;
end

function sim = simulate_one_mode(B, u, r, net_state, gps_state, par)
% Base standby
Pbase_scr = P_screen(0.0, par);
Pbase_cpu = P_cpu(par.u_idle, par);
Pbase_net = P_net('WIFI6', 0.0, par);
Pbase_gps = P_gps('off', par);
P_base_W = Pbase_scr + Pbase_cpu + Pbase_net + Pbase_gps;

% Mode absolute power
Pscr = P_screen(B, par);
Pcpu = P_cpu(u, par);
Pnet = P_net(net_state, r, par);
Pgps = P_gps(gps_state, par);

% Increment stacking
P_inc = (Pscr - Pbase_scr) + (Pcpu - Pbase_cpu) + (Pnet - Pbase_net) + (Pgps - Pbase_gps);
Pload_W = max(P_base_W + P_inc, 0.0);

% States
S = par.S0;
Vrc = 0.0;
T_C = par.T0_C;

dt = par.dt;
nmax = ceil(par.tmax_h*3600/dt);

t_arr = zeros(nmax,1);
S_arr = zeros(nmax,1);
Vt_arr = zeros(nmax,1);
Voc_arr = zeros(nmax,1);
IL_arr = zeros(nmax,1);
Vrc_arr = zeros(nmax,1);
T_arr = zeros(nmax,1);
P_arr = zeros(nmax,1);

k = 1;
for step = 1:nmax
if S <= par.S_min
break;
end

t = (step-1)*dt;

% RK4 integration for [S, Vrc, T]
x = [S; Vrc; T_C];
f = @(xx) deriv(xx, Pload_W, Pscr, Pcpu, Pnet, Pgps, par);

k1 = f(x);
k2 = f(x + 0.5*dt*k1);
k3 = f(x + 0.5*dt*k2);
k4 = f(x + dt*k3);
x_next = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);

S = min(max(x_next(1), 0.0), 1.0);
Vrc = x_next(2);
T_C = x_next(3);

% Record
[~, ~, ~, IL, Voc, Vt] = deriv(x, Pload_W, Pscr, Pcpu, Pnet, Pgps, par);

t_arr(k) = t;
S_arr(k) = S;
Vt_arr(k) = Vt;
Voc_arr(k) = Voc;
IL_arr(k) = IL;
Vrc_arr(k) = Vrc;
T_arr(k) = T_C;
P_arr(k) = Pload_W;
k = k + 1;
end

% trim
t_arr = t_arr(1:k-1);
S_arr = S_arr(1:k-1);
Vt_arr = Vt_arr(1:k-1);
Voc_arr = Voc_arr(1:k-1);
IL_arr = IL_arr(1:k-1);
Vrc_arr = Vrc_arr(1:k-1);
T_arr = T_arr(1:k-1);
P_arr = P_arr(1:k-1);

sim = struct();
sim.t_s = t_arr;
sim.t_h = t_arr/3600;
sim.S = S_arr;
sim.Vt = Vt_arr;
sim.Voc = Voc_arr;
sim.IL = IL_arr;
sim.Vrc = Vrc_arr;
sim.T_C = T_arr;
sim.Ptotal_W = P_arr;

sim.B = B; sim.u = u; sim.r = r;

sim.Pscr_W = Pscr;
sim.Pcpu_W = Pcpu;
sim.Pnet_W = Pnet;
sim.Pgps_W = Pgps;
sim.Pbase_W = P_base_W;
sim.Pload_W = Pload_W;
end

function [dx, dsdt, dVrcdt, dTdt, IL, Voc, Vt] = deriv(x, Pload_W, Pscr, Pcpu, Pnet, Pgps, par)
S = x(1);
Vrc = x(2);
T_C = x(3);

Qeff = Qeff_from_age(par);

% ECM params from SOC
[R0, R1, C1] = ecm_params_from_soc(S, par);
gamma = R1*C1;

% OCV
Voc = Voc_from_soc(S, T_C, par);

% Solve IL from power constraint
Vt = Voc - Vrc;
disc = Vt^2 - 4*R0*Pload_W;
if disc < 0, disc = 0; end
IL = (Vt - sqrt(disc)) / (2*R0);

% ODEs
dsdt = -IL / Qeff / 3600.0;
dVrcdt = (IL*R1 - Vrc) / gamma;

% Thermal
VL = Vt - IL*R0;
Ploss = (Voc - VL) * IL;

hA = par.h_Wm2K * par.A_m2;
Pheat = Ploss + par.eta_scr*Pscr + par.eta_cpu*Pcpu + par.eta_net*Pnet + par.eta_gps*Pgps;
dTdt = (Pheat - hA*(T_C - par.Tamb_C)) / par.Cth_JK;

dx = [dsdt; dVrcdt; dTdt];
end

function [p1, p2, lam1, lam2] = ocv_params_from_T(T_C, par)
Qage = par.Qage_Ah;

p1 = par.p1_c0 + par.p1_c1*T_C + par.p1_c2*Qage + ...
par.p1_c3*T_C^2 + par.p1_c4*T_C*Qage + par.p1_c5*Qage^2;

p2 = par.p2_c0 + par.p2_c1*T_C + par.p2_c2*T_C*Qage + ...
par.p2_c3*T_C^3 + par.p2_c4*T_C^2*Qage + par.p2_c5*T_C*Qage^2;

lam1_base = par.lam1_a1*T_C^2 + par.lam1_a2*T_C + par.lam1_a3;
lam2_base = par.lam2_b1*Qage^3 + par.lam2_b2*Qage^2 + par.lam2_b3*Qage + par.lam2_b4;

scale = par.Qref_Ah / par.Q0_Ah;
lam1 = lam1_base * scale;
lam2 = lam2_base * scale;
end
