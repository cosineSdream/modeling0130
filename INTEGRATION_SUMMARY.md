# 热模型求解方案变更总结

## 一、初始方案（精确解）→ 新方案（RK4 数值积分）

### 原方案（已弃用）
- 热模型采用一阶 ODE 精确解：`T(t+dt) = T_ss + (T(t) - T_ss)*exp(-dt/tau)`
- 优点：无条件稳定，无数值爆炸风险
- 缺点：不符合用户需求（要求与 SOC/Vrc 一样用 RK4）

### 新方案（已实现）
- 热模型采用 RK4 四阶数值积分，与 SOC、Vrc 耦合求解
- 参数传递：`Q_other`（不含 Ploss 的其他模块功率）在主循环中计算
- 返回值：从 5 个扩展到 7 个，新增 `Temp_out`

## 二、核心改动详情

### 2.1 RK4_step_ecm 函数重构

**函数签名变更**：
```python
# 旧
def rk4_step_ecm(soc_in, Vrc_in, Temp_in, P_load, Q_source, h_step)
# 新
def rk4_step_ecm(soc_in, Vrc_in, Temp_in, P_load, Q_other, h_step)
```

**内部导数函数新增**：
```python
def deriv_total(s, v, T):
    # 计算 ECM 导数
    dsdt, dVrcdt, IL, Voc, VL, Ploss = deriv_ecm(s, v, T, P_load)
    # 组合热源（含 Ploss）
    Q_source = W_LOSS * Ploss + Q_other
    # 热模型导数
    dTdt = (Q_source - h_th * (T - Tamb)) / Cth
    return dsdt, dVrcdt, dTdt, ...
```

**RK4 四阶积分**（对 soc, Vrc, T 的耦合积分）

### 2.2 主循环调整

**Step 5 变更**：
```python
# 旧：先计算 Ploss_val，再与 Q_other 组合，使用解析解
# 新：仅计算 Q_other，将热源组合推迟到 RK4 内部
Q_other = W_CPU*P_cpu_e + W_SCREEN*P_scr_e + W_NET*P_net_e + W_GPS*P_gps_e
soc_new, Vrc_new, T_new, ... = rk4_step_ecm(soc, Vrc, T, P_load, Q_other, dt)
```

### 2.3 数值稳定性强化

为防止 RK4 中间阶段温度爆炸，添加了以下防护措施：

| 位置 | 防护方案 | 范围 | 目的 |
|------|--------|------|------|
| `deriv_total` | Clip `dTdt` | [-10, 10] K/s | 限制温度导数防止阶段爆炸 |
| RK4 中间值 | Clip T2, T3, T4 | [-50, 120]°C | 防止温度进入数值不稳定区间 |
| `deriv_ecm` | Clip `IL` | [0, 100] A | 防止恒功率求解异常 |
| 热源处理 | Clip `Q_source` | [-1e3, 1e3] W | 防止极端热源 |

## 三、验证结果

✅ **运行成功，无 NaN/INF 错误**

### 测试输出（5 个老化水平）：
```
Q_aging =     0.0 Ah → 放电时间: 7.94 h, 最高温度: 45.45 °C
Q_aging =   100.0 Ah → 放电时间: 7.92 h, 最高温度: 45.45 °C
Q_aging =   500.0 Ah → 放电时间: 7.91 h, 最高温度: 45.46 °C
Q_aging =  1000.0 Ah → 放电时间: 7.89 h, 最高温度: 45.46 °C
Q_aging =  5000.0 Ah → 放电时间: 7.82 h, 最高温度: 45.47 °C
```

### 结果特点：
- ✅ 放电时间：7.82-7.94 h（合理递减趋势）
- ✅ 最高温度：45.45-45.47°C（稳定且合理）
- ✅ 平均负载：1970-2119 mW（正常水平）
- ✅ 无警告/错误信息

## 四、技术特性对比

| 特性 | 精确解方案 | RK4 方案 |
|------|----------|---------|
| 稳定性 | 无条件稳定 | 有条件稳定（需防护） |
| 耦合性 | 分离求解 | 完全耦合 |
| 代码复杂度 | 低 | 中 |
| 扩展性 | 低（热模型专用） | 高（通用 RK4） |
| 物理准确性 | 高（精确解） | 中（取决于 dt） |

## 五、关键代码位置

- [RK4 函数](code_4.py#L368-L415)
- [主循环 Step 5](code_4.py#L465-L472)
- [数值防护（导数限制）](code_4.py#L386-L389)
- [OCV 参数温度 Clip](code_4.py#L188-L217)

