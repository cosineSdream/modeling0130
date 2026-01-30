import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd

class SmartphoneBatteryModel:
    def __init__(self):
        # 电池参数
        self.Q0 = 4000.0          # 标称容量 (mAh)
        self.V_nom = 3.7          # 标称电压 (V)
        self.R0 = 0.1             # 参考内阻 (Ohm)
        self.Ea = 35000.0         # 活化能 (J/mol)
        self.R_gas = 8.314        # 气体常数
        self.T_ref = 298.15       # 参考温度 (K)
        self.alpha = 0.5          # 传递系数
        self.F = 96485.0          # 法拉第常数
        
        # 热参数
        self.C_th = 200.0         # 热容 (J/K)
        self.h = 10.0             # 传热系数 (W/m2K)
        self.A = 0.01             # 表面积 (m2)
        
        # 老化参数
        self.k_aging = 5e-7       # 老化速率常数 (h^-1)
        
        # 组件功耗参数
        self.P_screen_max = 0.6   # 屏幕最大功耗 (W)
        self.P_cpu_idle = 0.1     # CPU空闲功耗 (W)
        self.P_cpu_max = 3.0      # CPU最大功耗 (W)
        self.P_net_idle = 0.1     # 网络空闲功耗 (W)
        self.beta = 0.1           # 网络增量系数 (W/Mbps)
        self.P_gps = 0.1          # GPS功耗 (W)
        self.P_base = 0.05        # 基础功耗 (W)
        
    def V_oc(self, SOC):
        """开路电压曲线"""
        return 3.0 + 1.2*SOC - 0.4*SOC**2 + 0.1*SOC**3
    
    def R_int(self, SOC, T):
        """内阻模型"""
        R_base = self.R0 * (1 + 0.5/SOC)  # SOC越低内阻越大
        T_effect = np.exp(0.1*(1/T - 1/self.T_ref))  # 温度影响
        return R_base * T_effect
    
    def f_T(self, T):
        """温度容量修正因子"""
        return np.exp(self.Ea/self.R_gas * (1/self.T_ref - 1/T))
    
    def f_aging(self, t):
        """老化容量修正因子"""
        return np.sqrt(1 - self.k_aging * t)
    
    def Q_eff(self, T, t):
        """有效容量"""
        return self.Q0 * self.f_T(T) * self.f_aging(t)
    
    def component_power(self, t, scenario):
        """计算各组件功耗"""
        power = self.P_base
        
        # 屏幕功耗
        if 'screen_on' in scenario:
            brightness = scenario.get('brightness', 0.5)
            power += self.P_screen_max * brightness
        
        # CPU功耗
        if 'cpu_usage' in scenario:
            cpu_usage = scenario['cpu_usage']
            power += self.P_cpu_idle + (self.P_cpu_max - self.P_cpu_idle) * cpu_usage
        
        # 网络功耗
        if 'data_rate' in scenario:
            data_rate = scenario['data_rate']
            power += self.P_net_idle + self.beta * data_rate
        
        # GPS功耗
        if 'gps_on' in scenario:
            power += self.P_gps
        
        return power
    
    def solve_current(self, P_total, V_oc, R_int):
        """隐式求解电流"""
        # 迭代求解 I = P_total / (V_oc - I*R_int)
        I_guess = P_total / V_oc
        def f(I):
            return I - P_total / (V_oc - I*R_int)
        
        I = fsolve(f, I_guess)[0]
        return I
    
    def model_equations(self, t, y, scenario_func):
        """
        微分方程：dy/dt = f(t, y)
        y = [SOC, T_batt]
        """
        SOC, T_batt = y
        
        # 获取当前场景参数
        scenario = scenario_func(t)
        
        # 计算总功耗
        P_total = self.component_power(t, scenario)
        
        # 计算开路电压和内阻
        V_oc = self.V_oc(SOC)
        R_int = self.R_int(SOC, T_batt)
        
        # 求解电流
        I = self.solve_current(P_total, V_oc, R_int) * 1000  # 转换为mA
        
        # 有效容量
        Q_eff = self.Q_eff(T_batt, t)
        
        # SOC变化率
        dSOC_dt = -I / Q_eff
        
        # 温度变化率
        dT_dt = (I**2 * R_int / 1e6 - self.h * self.A * (T_batt - scenario.get('T_amb', 298.15))) / self.C_th
        
        return [dSOC_dt, dT_dt]
    
    def simulate(self, t_span, y0, scenario_func, max_step=60):
        """模拟电池放电"""
        sol = solve_ivp(
            lambda t, y: self.model_equations(t, y, scenario_func),
            t_span,
            y0,
            method='RK45',
            max_step=max_step,
            rtol=1e-6,
            atol=1e-9
        )
        return sol
    
    def find_empty_time(self, sol):
        """找到电池放空时间"""
        idx = np.where(sol.y[0] <= 0)[0]
        if len(idx) > 0:
            return sol.t[idx[0]]
        else:
            return sol.t[-1]

# 使用场景定义函数
def scenario_video_streaming(t):
    """视频流场景"""
    scenario = {
        'screen_on': True,
        'brightness': 0.7,
        'cpu_usage': 0.3,
        'data_rate': 2.0,  # Mbps
        'gps_on': False,
        'T_amb': 298.15  # 25°C
    }
    # 模拟周期性变化
    if t % 3600 < 1800:  # 每半小时休息5分钟
        scenario['screen_on'] = False
        scenario['cpu_usage'] = 0.1
    return scenario

def scenario_gaming(t):
    """游戏场景"""
    scenario = {
        'screen_on': True,
        'brightness': 1.0,
        'cpu_usage': 0.8,
        'data_rate': 0.1,
        'gps_on': False,
        'T_amb': 298.15
    }
    return scenario

def scenario_navigation(t):
    """导航场景"""
    scenario = {
        'screen_on': True,
        'brightness': 0.8,
        'cpu_usage': 0.4,
        'data_rate': 0.5,
        'gps_on': True,
        'T_amb': 298.15
    }
    return scenario

def scenario_cold_weather(t):
    """低温场景"""
    scenario = scenario_video_streaming(t)
    scenario['T_amb'] = 263.15  # -10°C
    return scenario

# 主程序
if __name__ == "__main__":
    # 初始化模型
    battery = SmartphoneBatteryModel()
    
    # 模拟参数
    t_span = (0, 24*3600)  # 24小时
    y0 = [1.0, 298.15]     # 初始SOC=100%, 温度=25°C
    
    # 不同场景模拟
    scenarios = {
        '视频流': scenario_video_streaming,
        '游戏': scenario_gaming,
        '导航': scenario_navigation,
        '低温视频流': scenario_cold_weather
    }
    
    results = {}
    
    print("开始模拟...")
    for name, scenario_func in scenarios.items():
        print(f"\n模拟场景: {name}")
        sol = battery.simulate(t_span, y0, scenario_func)
        empty_time = battery.find_empty_time(sol)
        results[name] = {
            'solution': sol,
            'empty_time_hours': empty_time/3600,
            'final_SOC': sol.y[0][-1]
        }
        print(f"  放空时间: {empty_time/3600:.2f} 小时")
        print(f"  最终SOC: {sol.y[0][-1]*100:.1f}%")
    
    # 敏感性分析
    print("\n=== 敏感性分析 ===")
    
    # 对屏幕亮度敏感性
    brightness_values = np.linspace(0.2, 1.0, 5)
    empty_times = []
    
    for b in brightness_values:
        def custom_scenario(t):
            s = scenario_video_streaming(t)
            s['brightness'] = b
            return s
        
        sol = battery.simulate(t_span, y0, custom_scenario)
        empty_time = battery.find_empty_time(sol)
        empty_times.append(empty_time/3600)
    
    sensitivity_brightness = np.gradient(empty_times, brightness_values)
    print(f"屏幕亮度敏感性: {np.mean(sensitivity_brightness):.3f} 小时/亮度单位")
    
    # 对温度敏感性
    temp_values = np.linspace(263, 313, 6)  # -10°C 到 40°C
    empty_times_temp = []
    
    for temp in temp_values:
        def temp_scenario(t):
            s = scenario_video_streaming(t)
            s['T_amb'] = temp
            return s
        
        sol = battery.simulate(t_span, y0, temp_scenario)
        empty_time = battery.find_empty_time(sol)
        empty_times_temp.append(empty_time/3600)
    
    sensitivity_temp = np.gradient(empty_times_temp, temp_values - 273.15)
    print(f"温度敏感性: {np.mean(sensitivity_temp):.3f} 小时/°C")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. SOC随时间变化
    ax = axes[0, 0]
    for name, result in results.items():
        sol = result['solution']
        ax.plot(sol.t/3600, sol.y[0]*100, label=name, linewidth=2)
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('SOC (%)')
    ax.set_title('不同场景下的SOC变化')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. 温度随时间变化
    ax = axes[0, 1]
    for name, result in results.items():
        sol = result['solution']
        ax.plot(sol.t/3600, sol.y[1] - 273.15, label=name, linewidth=2)
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('电池温度 (°C)')
    ax.set_title('电池温度变化')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. 放空时间对比
    ax = axes[0, 2]
    names = list(results.keys())
    times = [results[name]['empty_time_hours'] for name in names]
    bars = ax.bar(names, times)
    ax.set_ylabel('放空时间 (小时)')
    ax.set_title('不同场景放空时间对比')
    for bar, time in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{time:.1f}h', ha='center', va='bottom')
    
    # 4. 亮度敏感性
    ax = axes[1, 0]
    ax.plot(brightness_values, empty_times, 'o-', linewidth=2)
    ax.set_xlabel('屏幕亮度比例')
    ax.set_ylabel('放空时间 (小时)')
    ax.set_title('屏幕亮度敏感性分析')
    ax.grid(True, alpha=0.3)
    
    # 5. 温度敏感性
    ax = axes[1, 1]
    ax.plot(temp_values - 273.15, empty_times_temp, 's-', linewidth=2)
    ax.set_xlabel('环境温度 (°C)')
    ax.set_ylabel('放空时间 (小时)')
    ax.set_title('温度敏感性分析')
    ax.grid(True, alpha=0.3)
    
    # 6. 功耗分解（示例：视频流场景）
    ax = axes[1, 2]
    t_sample = np.linspace(0, 4*3600, 100)
    powers = []
    for t in t_sample:
        scenario = scenario_video_streaming(t)
        powers.append(battery.component_power(t, scenario))
    
    ax.plot(t_sample/3600, powers, linewidth=2)
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('总功耗 (W)')
    ax.set_title('视频流场景功耗变化')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('battery_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出详细结果表格
    print("\n=== 详细结果 ===")
    data = []
    for name, result in results.items():
        data.append({
            '场景': name,
            '放空时间 (小时)': f"{result['empty_time_hours']:.2f}",
            '最终SOC (%)': f"{result['final_SOC']*100:.1f}"
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # 保存参数到文件
    params = {
        'Q0': battery.Q0,
        'V_nom': battery.V_nom,
        'R0': battery.R0,
        'Ea': battery.Ea,
        'k_aging': battery.k_aging,
        'P_screen_max': battery.P_screen_max,
        'P_cpu_max': battery.P_cpu_max
    }
    
    import json
    with open('battery_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    print("\n参数已保存到 battery_params.json")
    print("模拟结果图已保存为 battery_simulation_results.png")

# 扩展功能：蒙特卡洛不确定性分析
def monte_carlo_analysis(battery, scenario_func, n_simulations=1000):
    """蒙特卡洛不确定性分析"""
    np.random.seed(42)
    
    empty_times = []
    
    for i in range(n_simulations):
        # 随机扰动参数（±10%）
        battery_temp = SmartphoneBatteryModel()
        
        # 随机扰动
        battery_temp.Q0 *= np.random.uniform(0.9, 1.1)
        battery_temp.R0 *= np.random.uniform(0.9, 1.1)
        battery_temp.P_screen_max *= np.random.uniform(0.9, 1.1)
        battery_temp.P_cpu_max *= np.random.uniform(0.9, 1.1)
        
        # 模拟
        sol = battery_temp.simulate((0, 24*3600), [1.0, 298.15], scenario_func)
        empty_time = battery_temp.find_empty_time(sol)
        empty_times.append(empty_time/3600)
    
    empty_times = np.array(empty_times)
    
    print("\n=== 蒙特卡洛分析结果 ===")
    print(f"模拟次数: {n_simulations}")
    print(f"平均放空时间: {np.mean(empty_times):.2f} 小时")
    print(f"标准差: {np.std(empty_times):.2f} 小时")
    print(f"95%置信区间: [{np.percentile(empty_times, 2.5):.2f}, "
          f"{np.percentile(empty_times, 97.5):.2f}] 小时")
    
    # 绘制分布图
    plt.figure(figsize=(10, 6))
    plt.hist(empty_times, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(empty_times), color='red', linestyle='--', 
                label=f'均值 = {np.mean(empty_times):.2f}h')
    plt.xlabel('放空时间 (小时)')
    plt.ylabel('频数')
    plt.title('蒙特卡洛模拟：放空时间分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('monte_carlo_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return empty_times

# 运行蒙特卡洛分析
print("\n运行蒙特卡洛不确定性分析...")
mc_results = monte_carlo_analysis(battery, scenario_video_streaming, n_simulations=500)

# 老化影响分析
print("\n=== 电池老化影响分析 ===")
years = np.arange(0, 5, 0.5)
initial_capacities = []

for year in years:
    hours = year * 365 * 24
    # 模拟老化后的初始容量
    battery_aged = SmartphoneBatteryModel()
    Q_eff_aged = battery_aged.Q0 * battery_aged.f_aging(hours)
    initial_capacities.append(Q_eff_aged)

plt.figure(figsize=(8, 5))
plt.plot(years, initial_capacities, 'o-', linewidth=2)
plt.xlabel('使用年限 (年)')
plt.ylabel('有效容量 (mAh)')
plt.title('电池老化对容量的影响')
plt.grid(True, alpha=0.3)
plt.savefig('aging_effect.png', dpi=300, bbox_inches='tight')
plt.show()

print("老化分析图已保存为 aging_effect.png")

# 导出完整报告
print("\n=== 生成分析报告 ===")
report = f"""
智能手机电池模型分析报告
生成时间: {pd.Timestamp.now()}
========================================

模型参数:
- 标称容量: {battery.Q0:.0f} mAh
- 标称电压: {battery.V_nom:.1f} V
- 参考内阻: {battery.R0:.3f} Ω
- 活化能: {battery.Ea:.0f} J/mol

主要结果:
"""
for name, result in results.items():
    report += f"- {name}: {result['empty_time_hours']:.2f} 小时\n"

report += f"""
敏感性分析:
- 屏幕亮度敏感性: {np.mean(sensitivity_brightness):.3f} 小时/亮度单位
- 温度敏感性: {np.mean(sensitivity_temp):.3f} 小时/°C

不确定性分析 (蒙特卡洛):
- 平均放空时间: {np.mean(mc_results):.2f} ± {np.std(mc_results):.2f} 小时
- 95%置信区间: [{np.percentile(mc_results, 2.5):.2f}, {np.percentile(mc_results, 97.5):.2f}] 小时

建议:
1. 降低屏幕亮度可显著延长续航
2. 避免高温或低温环境使用
3. 关闭不必要的后台应用
4. 根据使用场景选择合适的网络模式
"""

with open('battery_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("分析报告已保存为 battery_analysis_report.txt")
print("所有模拟完成！")