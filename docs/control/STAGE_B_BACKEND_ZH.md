> 历史说明：本文记录合并前的功能与验收过程；当前目录、命令及部署状态以 [统一项目说明](../../README_ZH.md) 为准。历史报告内容保留原有含义。

# B 阶段后端接入与隔离验证

2026-10-14。本轮完成关节执行后端、ROS 传输适配、QP 动作段准备和故障处理的代码接入。**未向真机发送指令，未部署到 8096 的运动入口，B 的现场验收尚未完成。**

## 已实现

- `prepared_joint_plan.py`：从新观察、原始预测调用已安装同核心 `NativeQP`，检查完整 50 步后提取指定前缀。保留夹爪原始值、限幅记录、原观察时间、关节顺序、2 ms QP 时序和文件哈希。求解后再次检查年龄，不给旧结果刷新时间戳。准备结果始终标记 `hardware_execution_ready=false`。
- `joint_execution.py`：`JointExecutionBackend` 接受关节命令封包，接入已有 `JointSourceGuard`。管理接管、独立心跳、关节限位/速度/时效、实际反馈跟踪、双臂逐侧发送、夹爪会话、保持确认和明确释放。不会把 ROS publish 返回解释为电机执行确认。
- `ros_execution_transport.py`：实际的 `JointcmdArm` User 输入发布、`set_input` 服务、输入选择读回，以及夹爪 command/heartbeat/status 通道。没有向共享 IK 输入添加发布者。读取真实关节位置/速度、原目标、机器人模式和夹爪底层反馈；检测 User 通道额外发布者。
- 生产域 24 的传输层只允许继承的静止关节目标，拒绝变化目标和夹爪 target 请求；必须先启动后台监测。没有通过配置布尔值开放模型运动的入口。
- `run.sh stage-b-sandbox` 提供可复现的隔离联调入口，固定域 231 和随机命名空间，不提供真机域参数，不启动机器人驱动/SDK。原 `run_control --mode hardware` 仍阻断模型运动；旧 TCP `CommandBridge` 保留为模拟接口，不假装它已具备 500 Hz 真机调度。

## 故障语义

1. 接管意图在发送前落盘。夹爪 claim 或输入切换回执丢失时不重发；重新读取当前所有权，只操作可确认属于本次会话的通道。
2. 任一提交失败会锁定故障；左右臂分别尝试保持最后已提交目标，任一侧失败不跳过另一侧；夹爪保持独立尝试。不会回退已经提交成功一侧的目标，也不会自动恢复遥操。
3. 用户心跳与指令提交分开；持续写指令不能续租。后台线程在调用方停止 tick 后仍检查租约、输入来源、模式和反馈。该线程与调用方共享进程，**不抵抗进程崩溃或操作系统卡死，不是独立硬件看门狗**。
4. 保持请求后，必须取得新鲜的新样本，持续满足静止速度、关节/夹爪跟踪及目标一致条件，才记录 `HOLD_CONFIRMED`。这仅是静止保持的反馈确认，`hardware_stop_verified` 始终为 false。
5. 未确认、所有权变化、过期反馈、驱动重启、日志故障等进入 `FAULT_UNCONFIRMED`，保留占用，不自动释放或继续旧动作。日志写入失败仍尽力请求保持。
6. 只有显式 `release` 且保持条件仍成立，才切回 Input=None、核对读回并释放夹爪。进程退出不自动释放未知状态。已有日志的后端实例启动为 `REVIEW_REQUIRED`，不会续接旧会话。

## 验证

- 完整集成目录回归：105 项测试通过；本轮新增 29 项（后端 25、QP 动作段 4）。包含安装版纯夹爪 Gate 的协议测试及实际 NativeQP 到关节封包的转换测试。
- 隔离联调使用已安装 `joint_cmd_mux` 二进制（3 秒来源过渡配置）和已安装纯夹爪 Gate，关节与电机反馈由理想模拟设备提供。
- 正常路径：User 双臂消息可见、夹爪请求确认、持续保持反馈确认、释放后 Input=None。
- 上游失联：后台监测触发保持，经模拟新反馈确认后显式释放。
- 右臂发送失败：左侧保持仍尝试，夹爪进入 held；最终 `FAULT_UNCONFIRMED`，保留会话，不伪称停止成功。

初次联调修正了两处接口适配：本机 rclpy Publisher 没有 `get_gid()` 方法，改用随机节点名/命名空间与发布者数量检查；重复选择已经选中的 Input 时 mux 不一定再次发布状态，改为结合服务成功回执和已有一致的锁存状态判定。来源变化仍要求切换后的新读回。ROS 图检查不是底层排他租约，也不能防止恶意发布者。

运行方式（只操作隔离域）：

```bash
cd /home/user/forcevla-web/robot_control
./run.sh test
./run.sh stage-b-sandbox --output runs/my-stage-b-normal
./run.sh stage-b-sandbox --output runs/my-stage-b-loss --fault upstream_loss
./run.sh stage-b-sandbox --output runs/my-stage-b-partial --fault right_write
```

## 仍需现场与后续工程验证

- 先完成 A 的真实静止验收，再为 B 接入显式现场启动流程。当前后端要求 Input=None，并取得足够新鲜的原关节目标及实际反馈；如果 None 后 mux 已停止输出，目标时间会过期，后端会拒绝，不能以实际位置冒充原目标。需要在接管流程中取得并核对原目标，或确认控制器可读目标接口。
- 尚未实测真机 User 来源切换、静止保持、夹爪会话、停止读回。隔离模拟的真实速度为零，不证明机器人动力学或运动中停止。
- 当前 B 接口按调用者提供的关节 tick 工作，日志落盘和夹爪回执等待不适合直接放入 500 Hz 动作循环。D 阶段需独立的有截止期调度、夹爪异步通道、跟踪/加速度/前缀停止约束；不得靠慢放轨迹绕过原 1/30 秒动作时序。
- 仅覆盖既有配置中的机器人碰撞对；环境、物体、完整夹爪、物理标定和接触执行仍待验证。
- 未修改在用驱动、mux、模型服务或网页服务，网页单次/连续推理仍为影子推理。

## 最终证据

- [交付摘要及源码哈希](../../data/integration-runs/stage-b-delivery-summary.json)
- [105项完整回归](../../data/integration-runs/stage-b-regression-final.log)
- [正常路径](../../data/integration-runs/stage-b-normal-final-v2/report.json)
- [单臂失败](../../data/integration-runs/stage-b-right-failure-final-v2/report.json)
- [上游失联](../../data/integration-runs/stage-b-upstream-loss-final-v2/report.json)

每个隔离运行目录保留源码快照、会话日志及mux日志。失败的早期适配记录也保留，以上链接为最终结果。
