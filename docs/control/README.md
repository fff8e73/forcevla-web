> 历史说明：本文记录合并前的功能与验收过程；当前目录、命令及部署状态以 [统一项目说明](../../README_ZH.md) 为准。历史报告内容保留原有含义。

# ForceVLA 真机验证：预检、只读采集与影子推理

> 2026-10-14 后续更新：B 阶段关节后端、ROS 适配和故障处理已实现并完成隔离验证；生产路径仍限制为静止目标，未部署/验收真机运动。见 [B 阶段后端说明](STAGE_B_BACKEND_ZH.md)。下文早期“真实写入未实现”描述仅反映当时状态，旧 TCP HardwareBackend 仍保持阻断。


本目录实现服务预检/预热、实时只读观察采集，以及步骤三的影子推理与离线QP核心诊断。使用现有推理服务和现有 ROS2 传感器，不启动/重启机器人服务，不创建手臂、夹爪或 IK 控制发布器。

`run_preflight.py` 会调用模型推理；`run_observe.py` 只采集数据，不调用模型。`run_predict.py` 调用现有服务并做离线IK，不向任何执行话题发布。已增加步骤四控制核心与模拟/就绪检查入口，真实硬件写入后端仍未实现。

## 1. 环境

本机已建立独立 `.venv`，使用系统 Python 3.12 的 ROS Jazzy、NumPy、SciPy、OpenCV、Pillow、PyYAML，并安装 `requirements.txt` 中的通信/解码依赖。没有改动部署模型的 venv。

复现安装：

```bash
cd /home/user/forcevla-web/robot_control
python3 -m venv --system-site-packages .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

`run.sh` 加载本机已有 ROS 环境，并使用独立 venv。每次运行使用新的输出目录，不覆盖旧报告。

## 2. 步骤一：服务预检与预热

```bash
cd /home/user/forcevla-web/robot_control
./run.sh preflight --output runs/my-preflight
```

检查内容：

- systemd 服务处于 active，RPC 端口归属于服务主进程，加载的启动配置指定 `reference-ordered` 和固定 checkpoint。
- 已部署清单与 `service-025` 验收清单相同；重新计算源码、资产及完整权重 SHA-256。
- 硬件模型/转换代码哈希和当前 ROS 参数符合 `config/robot.json` 中的审核基线；以 HTTP GET 保存机器人控制源、夹爪占用和标定快照。
- 检查30个冻结样本文件的完整性，使用 `sample_000` 的同一噪声进行一次初始请求和默认三次预热后请求。
- 每次预测通过原始归一化动作误差门槛，并与已接受部署输出以 `atol=1e-6, rtol=0` 对照；输出形状与有限值由客户端核验。

成功退出码为0，报告 `status=passed`。不通过则退出码1并保存原因，不自动改变参数、重启服务或放宽门槛。

输出包括 `report.json`、`parameters.json`、本次配置和 `frozen_prediction.npz`。记录的首请求是“本次调用的第一请求”，服务可能已被其他客户端预热，不能据此宣称测得冷启动。短程预热也不能替代完整模型验收或性能基准。

现有服务不提供实时运行清单 API，且引擎修改了进程标题；启动身份依据 systemd 配置、socket所属进程、磁盘清单和冻结输出交叉检查，不能描述成 RPC 已返回完整运行清单。

## 3. 步骤二：实时只读观察

先使用一小时内、相同配置的成功预检报告：

```bash
cd /home/user/forcevla-web/robot_control
./run.sh observe \
  --preflight-report runs/my-preflight/report.json \
  --output runs/my-observations \
  --duration 25 --hz 2 --max-samples 12
```

`--hz` 为保存快照频率，不改变传感器发布频率或训练动作步长。`--max-samples` 是本次目标数量，到达后提前结束；若 `--duration` 内未达到目标，退出码1并保留已有样本和拒绝原因。ROS接收、连续H.264解码和图像处理分别运行；解码后的帧队列只保留最新帧，避免昂贵图像处理阻塞压缩帧解码。保存文件时不会停止接收。

使用四路去畸变拼图、实际TCP、原始wrench及已归一化夹爪反馈。状态顺序为左 pose6/夹爪1/wrench6 后接右侧相同13维，输出为 `float32[26]`。只接受 base_link Pose、flange_L/R wrench、配置中声明的相机frame及电机端点。

以新相机帧为基准，从有界历史缓存中选择时间不晚于该帧的最新状态，检查源时间、接收时间、跨源偏差和底层电机反馈年龄。重复时间戳不刷新样本年龄；缺失、NaN、错frame、改标定和过期均记录拒绝，不填零或重复旧观察。

当前默认观察筛选预算：源/接收年龄0.5秒，跨源偏差0.15秒，夹爪底层反馈年龄0.2秒，未来时钟容忍0.02秒。这些是数据采集诊断预算，不是运动安全限值，不要直接用于控制阶段。

成功采集报告标记 `passed_readonly_capture`。该状态只表示获得了完整、符合配置筛选条件的观察，**不表示相机物理映射、TCP标定、力觉语义或真机控制验收通过**。`field_verification_pending` 保留未确认项。

### 输出文件

| 文件 | 内容 |
|---|---|
| `observation_*.npz` | 四路RGB uint8 HWC图像、原始26维状态和prompt，可作为后续推理输入 |
| `observations.jsonl` | 观察编号、文件哈希、状态、源/接收时间、偏差、解码时间与字段确认标志 |
| `camera_preview.png` | 第一份完整观察的四图：左上image、右上right_image、左下wrist_image、右下right_wrist_image |
| `report.json` | 采集数量、年龄/偏差摘要、拒绝计数、异常、清理结果 |
| `config.json` | 本次配置快照 |
| `source/` | 本次程序、测试及依赖文件快照 |

读取示例（仅读取已保存的数据）：

```python
import numpy as np
with np.load('runs/my-observations/observation_000000.npz', allow_pickle=False) as sample:
    request = {key: sample[key].copy() for key in (
        'image', 'right_image', 'wrist_image', 'right_wrist_image', 'state')}
    request['prompt'] = str(sample['prompt'])
    request['seed'] = 0
```

## 4. 相机预处理的已知边界

本次实采发现 `/quad_tile/compressed_undistorted` 是 H.264，解码1920×1488，而训练录制使用1920×1080。程序连续解码，保留packet/frame对应的源时间；中途订阅可能等待SPS/IDR关键帧，启动期间解码拒绝保存在计数中。解码队列有界，溢出后重置解码器并等待恢复，不用接收时间伪造源时间。

在线输入按显式配置先对完整拼图做AREA缩放到1920×1080，再复现转换器的四象限裁图、640×360 AREA缩放、上下60行黑边及JPEG95/4:4:4往返。该方案保留配置中的四视野与目标图像布局，但在线流与录制编码/缩放路径不同，不能宣称逐像素相同。配置中的 `camera_transport_equivalence_verified=false` 保留这一限制，仍需现场对照或同源在线/录制图像评估。

程序也保留 `camera_mapping_verified=false`、`physical_tcp_verified=false` 和 `wrench_physical_semantics_verified=false`。这些标志只在取得对应核验结果后更新；修改配置后必须重新预检。

## 5. 测试与故障定位

```bash
cd /home/user/forcevla-web/robot_control
./run.sh test -v
```

测试覆盖状态布局、夹爪不重复归一化、解码后帧积压不重置H.264参考链、旋转约定、错误frame、缺失/过期/乱序输入、因果时间对齐、底层反馈年龄、相机视图与补边、运行参数变化和原始数值门槛。

观察失败时先查看 `report.json` 的 `rejected_snapshots`、`diagnostics.stream_errors` 和各源最新时间。未收到关键帧可延长采集等待时长；传感器数据过期或配置变化应修复来源，不能默默放宽阈值或更换话题。不要运行控制SDK示例来尝试恢复采集。

## 6. 步骤三：实时影子推理与离线QP核心检查

已修复原生QP依赖兼容问题，默认使用独立原生求解器。首次构建先运行 `./build_native.sh`，它使用本目录固定的DAQP源码，未运行系统安装步骤。然后运行一次新预检（旧配置哈希的报告不能复用）：

```bash
cd /home/user/forcevla-web/robot_control
./run.sh preflight --output runs/my-step3-preflight
./run.sh shadow \
  --preflight-report runs/my-step3-preflight/report.json \
  --output runs/my-shadow \
  --predictions 3 --duration 90 --seed 0
```

采集仍在独立线程持续运行。影子模式额外只读订阅 `/tj/joint_states`，按 `Joint1_L…Joint7_L,Joint1_R…Joint7_R` 名称重排并做与相机一致的因果时间筛选，作为每个动作段的实测IK初值。缺少关节反馈、过期或名称重复均拒绝，不猜测零位。模型输入仍是原有26维状态，不把关节角加入模型请求。

每次观察生成UUID，用作gRPC request_id并检查响应ID；保存观察、显式seed、原始预测及处理时间。整个50步动作检查包含：

- 形状、有限值、观察至响应年龄；相对当前TCP的位移、相对旋转角、逐步变化、夹爪范围。越界值原样记录，不裁剪后宣称通过。
- 以实际关节角做FK，与同期实际TCP比较，防止错误构型或坐标链进入求解。
- 使用完整 `m6_696.xml` 和已安装的 `QpControllerCore`，按当前500 Hz参数计算50个30 Hz目标，共834个控制器周期；初值来自同期实际关节反馈。
- 每个计算周期检查原生QP成功状态、关节限位和速度；每个目标间隔末端检查有限时间跟踪误差和关节连续性。
- 每个QP计算周期独立检查当前配置中的8对机器人碰撞几何与5 mm间距。它不是外部环境碰撞或连续时间碰撞证明；在线包装层和平滑器也不在此数值预演中。

诊断参数在 `config/planning.json` 固定并进入每轮报告，包括：响应年龄2.5秒、相对TCP位移0.15米/旋转0.6 rad、每步0.03米/0.25 rad；IK位置残差0.01米/旋转0.1 rad、每步最大关节变化0.15 rad。上述数值是本轮影子排查预算，不是用户批准的运动限值，也不是任务成功标准。不得为了消除失败记录而静默放宽。

### 离线求解器与在线控制链的区别

原先QP静态核心与Python DAQP轮子存在结构布局不兼容，自检在 `solve_qp_daqp` 崩溃。本轮用DAQP官方v0.7.2固定提交独立静态编译，112字节设置布局通过编译断言和保持/运动/异常测试，恢复了同核心离线计算。构建入口 `build_native.sh` 拒绝未固定或被修改的源码，输出只在本目录；没有替换在线控制器或系统库。源码来源与MIT许可见 [vendor说明](vendor/README.md)。

`native_qp.py` 调用 `build/offline_qp` 独立子进程，后者只加载数值核心，不初始化ROS、不连接SDK。报告标记 `same_installed_qp_core=true`。`offline_ik.py` 的通用几何IK保留用于对照，默认影子入口不再以它替代QP；原生构建缺失或哈希变化时失败，不静默回退。

同核心不等于完整在线控制链：在线DAQP原始精确源码版本、ROS包装层、平滑器、mux、网络和机器人动态仍未完成逐项对照，`online_qp_equivalence_verified=false`。这些差异及限度必须保留。

现有配置仅覆盖部分手臂/底座碰撞对，未包含桌面、工装、插头和实物夹爪几何。因此 `environment_collision_checked=false`，任何结果的 `hardware_execution_ready` 都保持false。进入步骤四前仍须完善场景、验证控制权和真实停止行为。

### 输出与退出状态

每次输出目录包含配置、源码快照、`solver_manifest.json` 和总 `report.json`。每个 `prediction_XXXX/` 保存：

| 文件 | 内容 |
|---|---|
| `observation.npz`、`observation_metadata.json` | 四图、26维状态、14维实测关节初值、prompt、seed和时间戳 |
| `prediction.npz` | 完整物理与归一化预测，未改动夹爪目标 |
| `action_checks.json` | 逐目标TCP/夹爪/时效诊断 |
| `ik_checks.json` | 初值FK误差、QP状态、逐目标跟踪误差、周期速度/碰撞检查与范围限制 |
| `qp_trajectory.npz` | 原生QP的834周期关节轨迹与对应目标编号；初值校验失败时不生成 |
| `qp_input.json`、`qp_output.yaml` | 数值子进程输入与原始结果，无硬件发布 |
| `report.json` | 观察/request ID、RPC耗时、拒绝原因、文件哈希和分析结束时数据年龄 |

`--duration` 是开始新请求的采集窗口，已发出的RPC/离线求解分别受自身超时限制；到达请求次数后提前退出。不足请求数、RPC错误或程序异常返回退出码1，保留失败记录。

所有请求成功得到诊断时退出码0，总状态为 `completed_shadow_run`。这只表示流程完成：必须继续查看 `diagnostic_assessment`、逐次 `action_reasons`、`ik_reasons` 和 `post_analysis_reasons`。模型返回后分析仍在消耗原观察的年龄，超预算会标记analysis_result_stale。包含拒绝时仍保存全部分析，但绝不代表“允许执行”。即便全部诊断通过，也不跨过尚未验证的物理和控制器条件。

早期结果见[步骤三验证记录](PREDICTION_VALIDATION_ZH.md)，最新同核心QP修复、夹爪审计与实测结果见[后续问题定位记录](CONTINUATION_VALIDATION_ZH.md)。

## 7. 步骤四软件验证与硬件就绪检查

已实现 `command_bridge.py`、`hardware_backend.py`、`run_control.py`：包含有限段状态机、等待保持、控制权/反馈/时效检查、纯ROS消息转换及理想化模拟故障验证。真实传输和实际停机行为尚未验证，硬件入口固定阻断，不能用于现场动作。

```bash
./run.sh control --mode simulate --output runs/my-control-sim
./run.sh control --mode audit --output runs/my-hardware-readiness
```

模拟使用独立合成数据和虚拟时间；audit只做HTTP GET及ROS图读取。即使使用 `--mode hardware` 也只会检查并报告阻塞原因，不创建运动发布器。当前不能通过修改布尔配置解锁硬件写入。

详见[步骤四实现、测试与阻塞报告](CONTROL_VALIDATION_ZH.md)。下一步需解决步骤三拒绝和现场语义/场景、在线QP等价性以及独占占用和实际停止协议，再完成真实后端和小范围动作验证。

关联：[实施计划](../forcevla-control-planning/IMPLEMENTATION_PLAN_ZH.md)、[硬件核查](../forcevla-control-planning/HARDWARE_SDK_AUDIT_ZH.md)。

## 8. 自定义user输入候选与门禁

已确认现有mux支持网页未显示的Input=3，完成独立ROS域下的二进制路由与旧指令实验；新增进程内租约/序号/时效门禁，尚未接入在用控制链。详情见[输入来源验证](INPUT_SOURCE_VALIDATION_ZH.md)。不要把新增门禁视为真实后端已启用。

夹爪接管判断更正：实际负反馈不必然阻止claim，真实流程继承现有目标并检查速度/跟踪容差，详见[勘误](GRIPPER_CLAIM_CORRECTION_ZH.md)。

## 分段闭环 V1 软件

已增加统一参数、受约束前缀准备、原生定时执行、多轮编排及8096页面入口。使用方式与软件/真机验收边界见[分段闭环说明](CLOSED_LOOP_V1_ZH.md)。`run.sh closed-loop --mode simulate`使用合成观察与设备；真机入口要求M6实测资格资料，旧静止验证入口不会因此开放运动。
