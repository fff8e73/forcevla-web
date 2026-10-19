> 历史说明：本文记录合并前的功能与验收过程；当前目录、命令及部署状态以 [统一项目说明](../../README_ZH.md) 为准。历史报告内容保留原有含义。

# 阶段C/D本轮尝试（2026-10-14）

C完成新观察、真实模型推理和同核心QP前缀准备；D在运动预算检查时拒绝，未发送模型动作。

- 服务、模型文件、参数与冻结样本检查通过。初次ROS参数查询出现节点发现超时；改用独立发现并等待3秒后通过，未重启机器人或模型服务。
- 真机数据推理约1.397秒；原始观察至C准备完成2.392秒，满足当时2.5秒门槛，但余量仅约0.108秒。
- 选择模型第1步，得到17个2ms关节目标，共34ms；没有慢放、修改原观察时间或重用旧预测。
- 最大关节速度9.01039rad/s，超过首轮0.1rad/s；最大关节变化0.02209rad，超过0.01rad；包含起止衔接的加速度4738.59rad/s²，超过1.0rad/s²；TCP位移6.463mm超过5mm，旋转0.03037rad略超0.03rad。
- 夹爪目标前缀无变化；本轮拒绝与此前夹爪外层速度监测阈值无关。
- D还需要约束动作生成/时序、完成500Hz调度验收、明确现场运动预算及独立停止路径。B刚完成的正常固定目标保持已作为前置证据接入，不再误报缺少这一证据。

保存结果仅供诊断，现在不能执行；后续修改需从新观察重新推理和检查。已有B真机结果不等于D模型动作或E运动中停止通过。

证据：[C/D摘要](../../data/integration-runs/stage-cd-after-hold-prepare-01/attempt-summary.json)、[动作预算](../../data/integration-runs/stage-cd-after-hold-prepare-01/stage_d_budget_checks.json)、[完整准备记录](../../data/integration-runs/stage-cd-after-hold-prepare-01/report.json)、[预检](../../data/integration-runs/stage-cd-after-hold-preflight-02/report.json)。


## 按用户要求诊断新预测前5步

已重新采集真机数据并推理，非旧预测复用。C准备时观察年龄2.244秒，5个模型动作对应84个2ms关节目标、共0.168秒。

检查拒绝：最大关节速度8.39944rad/s（首轮上限0.1）、关节位移0.04791rad（上限0.01）、加速度含停止边界4521.59rad/s²（上限1）、TCP位移11.386mm（上限5）、旋转0.05120rad（上限0.03）。夹爪目标无变化。

没有下发真机运动指令。只减少动作数量无法消除起步瞬间的速度/加速度超限。该5步入口仅用于诊断，不能解锁实际执行器，相关15项前缀检查测试通过。所有后续执行都必须从新观察重新准备，不执行这里已保存的旧结果。

证据：[前5步尝试摘要](../../data/integration-runs/stage-d-five-live-observation-01/attempt-summary.json)、[预算检查](../../data/integration-runs/stage-d-five-live-observation-01/stage_d_budget_checks.json)。
