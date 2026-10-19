> 历史说明：本文记录合并前的功能与验收过程；当前目录、命令及部署状态以 [统一项目说明](../../README_ZH.md) 为准。历史报告内容保留原有含义。

# 实测位置固定保持：实现与尝试结果

2026-10-14。用户授权将当前实测关节角作为新的固定保持目标，验证B关节通道。该方案不要求原目标读回，也不宣称继承原控制器目标不变。

- 实现：`measured_hold.py`；入口：`diagnostics/measured_hold_probe.py`。默认只读预检，显式 `--live` 才会启动已授权的静止控制流程。
- 只允许Position模式、None或空闲Planner；先采集连续静止反馈，切None后采样一次固定目标。不会不断跟随反馈改目标。
- 夹爪目标保持原值；生产传输仍拒绝目标变化。持续读取真实mux输出用于确认，不用实测角度伪造目标回执。
- 缺失mux目标、位移/速度异常、陈旧反馈、所有权变化等不能判定成功；不自动释放未确认会话。

## 结果

隔离域231验证通过：67对固定关节目标发送，模拟设备保持目标，完成保持确认及释放。未启动硬件驱动。

真机尝试停在预检：第一次超过0.02rad/s；核对已通过的A阶段配置后，夹爪持续监测采用既有0.03rad/s标准（双臂仍0.02，驱动claim瞬间仍0.02）。第二次右夹爪速度为−0.036630038rad/s，仍超限。没有修改阈值放行，也没有切Input、claim、发布关节指令或夹爪目标。

这只说明采样时夹爪未满足本测试的静止门槛，不足以诊断电机故障或确定其物理运动幅度。需要检查实际静止情况，再做新一轮采样。

证据：

- [隔离流程](../../data/integration-runs/measured-hold-sandbox-01/report.json)
- [首次真机预检](../../data/integration-runs/measured-hold-live-preflight-01/report.json)
- [按A阈值复查](../../data/integration-runs/measured-hold-live-preflight-02/report.json)
- [完整回归](../../data/integration-runs/measured-hold-regression-final.log)

B真机通道验收、D真机运动及E运动中停止均未据此标记通过。


## 后续真实固定位置保持：通过（2026-10-14T20:31:32+08:00）

用户明确要求将夹爪外层速度监测阈值调整为0.5rad/s，并观察固定位置保持。此次通过独立命令参数启用，默认值及在用驱动不变；没有修改驱动claim瞬间0.02rad/s限制。双臂仍要求速度≤0.02rad/s、相对固定目标偏移≤0.005rad；新增夹爪相对接管时位置偏移≤0.01rad限制。

已实际执行：Planner→None→采集一次新关节目标→夹爪继承原目标claim→User通道固定目标保持约4.2秒→hold及真实反馈确认→None→夹爪release。

- 固定目标保持循环发送 **67对双臂指令**；加上结束保持指令，左右各93条ROS关节消息。不是模型动作。
- 采样最大关节目标偏差 **0.000000932106rad**，最大关节速度 **0.001868341rad/s**。这些是记录样本的统计，不声称连续时间零位移。
- 采样最大夹爪速度 **0.021978023rad/s**；夹爪目标始终 `[0,0]`。
- 保持后观测到连续约0.2秒的新鲜反馈满足本轮门槛，随后释放。
- 独立只读复核：Robot Mode=Position、Input=None、control_ready=true、control_error=false、夹爪idle且reserved=false。

这完成了B的**正常固定目标真机关节通道接入、保持与释放**，不是原目标无扰继承、运动中停止或模型任务验收。B异常注入仍只有隔离证据；不因此开放D/E/F。

原始报告中的 `stationary_feedback_confirmed=true` 只反映该版本结束时样本判定，不能证明放宽阈值后的整个夹爪静止过程。审计摘要明确不将其作为原静止标准通过；后续代码在外层阈值>0.03时对此字段保守返回false。原始报告和源代码快照保留不改。

证据：[实测完整报告](../../data/integration-runs/measured-hold-live-monitor05-01/report.json)、[审计摘要](../../data/integration-runs/measured-hold-live-monitor05-01/acceptance-summary.json)、[独立结束读回](../../data/integration-runs/measured-hold-live-monitor05-01/independent-readback-summary.json)、[反馈采样](../../data/integration-runs/measured-hold-live-monitor05-01/observations.json)。新增范围及完整回归见[最终测试日志](../../data/integration-runs/measured-hold-monitor-override-tests-final.log)。
