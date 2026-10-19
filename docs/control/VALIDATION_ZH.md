> 历史说明：本文记录合并前的功能与验收过程；当前目录、命令及部署状态以 [统一项目说明](../../README_ZH.md) 为准。历史报告内容保留原有含义。

# 步骤一、二实施与验证记录

日期：2026-10-14。范围为现有推理服务预检和真实只读观察采集，不含影子IK或动作控制。

## 最终结果

- [最终预检报告](../../data/integration-runs/20261014-preflight-final/report.json)：passed。源码/资产/完整权重哈希、服务配置、模型参数和夹爪标定核验通过；冻结样本一次初始请求加三次后续请求，原始门槛及已接受输出对照均通过。预热后RPC中位数 1257.54 ms，仅是本次短程检查。
- [最终采集报告](../../data/integration-runs/20261014-observe-final/report.json)：passed_readonly_capture，目标20份、实际20份。最大源年龄 175.74 ms，最大跨源偏差 51.62 ms。
- [四图预览](../../data/integration-runs/20261014-observe-final/camera_preview.png)：四路均有实际画面，保留配置视图顺序及黑边；物理相机身份未凭画面自动确认。
- 独立读取验证20个NPZ：共80个640×480 RGB uint8图像，20个有限float32[26]状态，样本SHA-256全部匹配；运行源码快照与报告及当前实现一致。
- 16项单元/回归测试通过；Ruff F规则检查通过。覆盖字段、旋转、夹爪、时间拒绝、因果对齐、图像布局、运行参数变化和解码后帧积压。
- 当前实现没有应用层控制发布器或硬件SDK会话。仅冻结样本RPC、HTTP GET、ROS参数读取与传感器订阅；观察入口不执行推理。

## 实施中发现并修正的问题

1. 模型进程改写标题，不能从/proc命令行读原始flags。改为核对systemd ExecStart、socket所属PID、运行目录与冻结结果；服务没有实时manifest RPC，该证据限制保留在报告。
2. 在线去畸变流为H.264、1920×1488，训练录制拼图为1920×1080。最初运行因尺寸不符拒绝；后续显式配置和转换几何，并记录在线/训练编码路径等价性尚未验证。
3. 图像预处理阻塞压缩帧消费，产生队列溢出。改为独立接收、持续解码和最新解码帧处理，避免为丢弃旧画面而破坏H.264参考链。
4. 第一轮服务调用曾耗时41.35秒，单独记录，后续请求约1.2秒；没有将第一请求算入稳态统计，也未宣称测得冷启动。
5. 早期诊断运行observe-002只采到2份数据；最终实现要求达到指定样本目标才成功，未达到时保留部分文件并返回失败。早期报告保留用于排查，不作为最终验收。

最终采集拒绝计数：`{"missing:left_pose,right_pose,left_wrench,right_wrench,gripper,camera": 1, "invalid_stream:camera": 5, "stale_motor_feedback": 3, "no_causal_sample:left_pose": 6}`。启动缺流或等待关键帧仍会被拒绝；不会用接收时间、零值或旧样本补齐。

## 尚未通过的现场条件

camera_mapping_verified、camera_transport_equivalence_verified、physical_tcp_verified、wrench_physical_semantics_verified 均为false。当前通过的是软件预检与带时效筛选的只读采集，不是完整P1现场语义验收，更不是任务控制验收。配置中的筛选预算也不是运动安全限值。

操作见[README](README.md)。本轮没有实现或运行步骤三及之后的控制逻辑。
