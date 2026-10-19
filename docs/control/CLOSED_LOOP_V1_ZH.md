> 历史说明：本文记录合并前的功能与验收过程；当前目录、命令及部署状态以 [统一项目说明](../../README_ZH.md) 为准。历史报告内容保留原有含义。

**分段闭环 V1：软件运行与验证说明**

流程：保持确认 → 新观察 → 推理50步 → 准备并执行前K步 → 保持确认 → 下一轮。默认K=3、最多3轮、总时长30秒。达到轮数标记`COMPLETED_LIMIT`，不会标记插接任务成功。

模型输出间隔为1/30秒。关节轨迹在原时间网格上生成，每2ms提交一次。轨迹准备以新观察作为模型和参考IK输入，同时从已经确认的固定指令目标平滑起步；实际位置与该目标的偏差必须符合起点预算。新模型结果不会从第50步继续，也不会复用上一轮未执行的尾段。

**构建与运行**

```bash
cd /home/user/forcevla-web/robot_control
./build_loop.sh
./run.sh closed-loop --mode simulate --output runs/my-new-closed-loop
```

每次指定新的输出目录。模拟模式使用合成图像、合成模型、简化几何和原生执行组件内部的理想设备；不初始化该组件的ROS发布器，不发送真机指令。它可以验证反馈闭环和时序，不能证明实际运动效果。

`config/closed_loop_v1.json`集中配置K、轮数、超时、时效、速度/加速度、跟踪及保持条件。K当前范围1～3；修改配置后需要重新核对相关验收。CLI提供`--cycles`、`--prompt`、`--seed`；网页提供有限轮次和时长设置。

真机模式使用同一编排器和原生定时组件，但必须通过当前代码、配置、模型声明、数值资产和实测资料的资格核对。当前未生成M6资格资料，真机入口会拒绝启动。仅修改`ready`或`sandbox`标志不能开放执行。M6的具体试验与现场参数需另行实施，本文的软件验收不代替它。

**组件职责**

- `closed_loop_contracts.py`：配置、观察、动作段、时间屏障、身份及取消契约。
- `closed_loop_observation.py`：检查图像、状态、关节、夹爪源时间及底层反馈年龄均越过保持屏障；保存原采样时间。
- `closed_loop_planning.py`：前缀QP参考、受约束轨迹、TCP/碰撞复核及各中断点制动路径。未改变模型的30Hz时间语义。
- `closed_loop.py`：有限轮次状态机、隔离计算进程、预热、超时、取消、日志和结束分类。
- `closed_loop_native.py`与`native_loop/loop_motion.cpp`：私有socketpair、原生500Hz发送、独立反馈及监督租约检查、取消与已检查制动路径。正常保持允许下一轮，终止状态不允许恢复。
- `closed_loop_execution.py`：控制会话、夹爪独立通道、正常保持及最终停止反馈确认。Python定时路径保留用于诊断；V1 CLI默认使用原生定时路径。
- `closed_loop_live.py`：真实ROS适配、只读反馈缓存、资格和模型/资产身份核对。旧Stage B静止适配器仍只允许固定目标。
- `forcevla-web/forcevla_web/closed_loop.py`：独立工作进程管理、会话归属、幂等、互斥、历史和导出。浏览器不承担控制节拍。

**保持、取消与故障**

正常段结束维持固定目标并确认连续新反馈；控制会话保留。当前User输入切换包含3秒渐变，真实ROS路径在首次接管后额外等待已配置的3.2秒，确认后才开始第一轮观察。

停止请求会使当前计算结果失效。执行中使用预先检查的制动路径，之后确认双臂与夹爪保持；推理/准备期间直接终止推进并保持。夹爪在途回执不得重新激活已经终止的会话。停止请求回执、指令发送记录和实际保持确认分别记录。

Python编排器持续提供带原始时间的监督信号。原生组件即使仍收到状态轮询，也不会把旧监督时间续期；Python进程退出导致私有连接断开时，原生组件请求停止并保存已有记录。上游失联或设备反馈异常不会自动恢复旧轨迹。机器物理停止效果仍需M6确认。

未确认停止时保留不确定状态；网页阻止新会话，不把进程消失解释为已停稳。服务重启不会自动恢复任何动作。正常结束确认保持后释放，本会话将Input置为None，不自动恢复遥操。

**网页接口**

- `GET /api/v1/closed-loop/status`
- `POST /api/v1/closed-loop/runs`
- `GET /api/v1/closed-loop/runs`
- `GET /api/v1/closed-loop/runs/{id}`
- `POST /api/v1/closed-loop/runs/{id}/stop`
- `GET /api/v1/closed-loop/runs/{id}/export`

启动请求包含`mode`、`prompt`、`seed`、`execution_horizon`、`max_cycles`、`max_task_duration_s`和`command_id`。同一标识重试返回同一运行；变更参数不能复用旧标识。启动/停止继承网页的会话、同源与CSRF校验。闭环与影子推理、静止控制验证互斥。

页面刷新仅恢复展示。浏览器本会话失联超过配置预算会取消运行。停止按钮显示“请求已发送”后，继续等待实际保持结果。模拟闭环有明确标识，上方实时相机不代表模拟设备。

**记录与复核**

每个运行保存`manifest.json`、`status.json`、`events.jsonl`、执行日志和各轮`cycles/0001/…`。每轮包含新观察、原始50步预测、夹爪适配、原生QP参考、受约束轨迹及制动分支、检查结果和实际结束时间。

`native/native-ticks.jsonl`保存实际发送的目标、计划编号、计划截止时间、发送起止时间、实际关节位置/速度、反馈时间及双臂提交位。`kind=1`为模型前缀，`kind=2`为制动，`kind=0`为固定保持；固定保持不伪装成新的模型预测。`native-result.json`保留发送统计和组件结束状态。

模型身份记录绑定部署清单声明及数值资产，不把gRPC健康返回当作完整模型加载身份。M6还需完成现有部署预检和相应硬件实测。日志容量有界，记录失败不应阻止停止请求；证据缺失时不报告完整通过。

原生定时线程在启动时采样150ms，选择进程可用CPU中当时负载最低的一个并固定，将定时器slack设为1ns，在固定保持期间于截止前400µs进入短等待，实时调度下短动作段同样保留每周期约1.6ms的睡眠；普通调度的诊断回退路径在短动作段内持续检查绝对截止时间；优先使用直接允许的FIFO优先级10，否则向机器已有的RealtimeKit申请优先级10；设置200ms连续实时CPU时间上限，记录申请结果和实际FIFO/RR策略。没有修改主机全局调度设置。实际CPU、调度策略和每次迟到值保存在原生报告中；这些设置不构成硬实时保证，超期仍停止。

**验证入口**

```bash
./run.sh test -v
./run.sh closed-loop --mode simulate --output runs/new-simulation
source /opt/kernelmind/apex/install/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python diagnostics/closed_loop_ros_sandbox.py --output runs/new-ros-isolated
```

ROS诊断固定domain=231和随机`/forcevla_sandbox_…`命名空间；使用已安装mux、纯夹爪Gate及独立的合成设备进程，不启动机器人驱动。三轮每轮50个关节周期，共150次双臂目标提交；其模型输入和设备动力学仍是模拟。

网页回归在`/home/user/forcevla-web`执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python tests/closed_loop_browser_server.py
# 另一个终端：
node tests/browser_closed_loop.mjs
```

最终验收记录见规划目录中的M1–M5实现报告。M6真机运动、运动中停止和任务成功率保持单独验收。
