> 历史说明：本文记录合并前的功能与验收过程；当前目录、命令及部署状态以 [统一项目说明](../../README_ZH.md) 为准。历史报告内容保留原有含义。

# 指定前 10 步的真机执行

本次已完成：原记录 `20261014-211413-CURRENT_OBSERVATION_FIRST10_ZH.md` 的 10 个双臂目标均在真机逐点确认到位。成功记录位于 `runs/first10-planner-live-05/`。

- 10 点运动及到位等待共约 7.005 秒；采用原生 MoveJ 重新定时，非原始 30 Hz 回放。
- 每个关节的规划速度上限 0.2 rad/s、加速度上限 0.5 rad/s²，来自运行中 `/tj/planner_joint_node` 的参数读取。没有修改这些参数，也不将它们称为硬件额定上限。
- 逐点实测 TCP 相对原始目标的最大位置误差 0.475 mm、最大姿态误差 0.00735 rad。
- 结束时已通过实际反馈确认双臂静止、Input=None，夹爪控制权释放。两侧夹爪目标始终为 0。
- 按用户指示删除自行添加的原始力变化 10 N / 力矩变化 1 N·m 停止条件。原始力/力矩和相对启动值的变化仅记录，不作为未经标定的接触判断。
- 本结果验证这 10 个目标的执行，不作网线插接任务成功判定。

## 入口

```bash
cd /home/user/forcevla-web/robot_control
./run.sh first10                  # 当前起点核对、离线准备；不发送控制指令
./run.sh first10 --execute        # 明确下发真机动作
```

默认起点必须与原记录一致。成功执行后当前位置已经改变，再次运行将受起点核对约束。

代码：`run_first10.py`（ROS、控制权、运行记录）、`first10_replay.py`（原始文件校验、QP 准备、逐点执行与停止）。

本次中途停在第 1 点之前后，使用 `--resume-stopped-run runs/first10-planner-live-04` 显式继续。该入口仅接受相同来源、尚未完成任何目标、已经确认静止并释放控制权的记录，且重新核对当前关节/TCP 与该停点一致。保留原预测时间，记录为指定目标复现，不把旧预测伪装成新推理。

## 执行与验证

读取 NPZ 精确 float32 目标及原观察，校验 SHA-256，并与完整 50 步输出的前 10 行核对。仅将夹爪目标显式裁剪到 [0,1]；本次 10 行均为 0，与现有目标一致。

使用完整 m6_696 模型和已安装 QP 核心离线求解每个目标的关节终点，再通过既有 `/tj/control/movej` 和 Input=2 规划器通路执行。起点先在 Input=None 下初始化规划器；选源后再次提交同一静止目标，使 mux 收到新的样本。每点检查新鲜实际反馈、关节和 TCP 到位；异常停止后不继续消费后续目标。

原生规划器停止发布已完成的轨迹后，保留最后目标的原始时间，使用持续更新的实际位置/速度确认静止。没有把历史命令的时间刷新为当前时间。

验证包含 10 项单元测试、只读真机预检、隔离 ROS 域 231 中已安装规划器的完整 10 点执行及配置碰撞对检查、真机逐点实际反馈，以及结束后的独立 HTTP 读回。配置碰撞几何不覆盖桌面、夹具或完整环境。

记录：

- `runs/first10-native-planner-rehearsal-03/report.json`：原生规划器隔离验证。
- `runs/first10-planner-live-01/` 至 `04/`：保留早先拒绝、接入诊断及部分运动停止记录。
- `runs/first10-planner-live-05/report.json`：成功运行及控制权/停止回执。
- `runs/first10-planner-live-05/acceptance_summary.json`：逐点 TCP 误差和独立结束状态。
- `runs/first10-planner-live-05/completed_steps.json`、`feedback.json`、`events.jsonl`：逐点实际读回、时序与原始力/力矩。
- `runs/first10-planner-live-05/final_preview.png`：结束后的四路画面。
