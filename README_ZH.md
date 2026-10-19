# ForceVLA Web：推理与机器人控制

统一项目目录为 `/home/user/forcevla-web`。网页、观察采集、推理调度和原控制集成代码现在放在同一个项目中，由 `forcevla-web.service` 提供 8096 页面。真机控制仍在独立子进程运行；网页、ROS 采集和控制各用自己的 Python 环境。

## 页面功能

- **开始连续**：读取当前观察，推理，执行前 K 步，等待新观察后进入下一轮。默认 50 轮、每轮 20 步，前端可调整轮数和步数；总时长上限仍为 600 秒。
- **停止真机闭环**：停止当前真机循环。
- **仅单次推理**：生成和展示预测，不执行动作；**停止推理**结束此类推理会话。
- **HOME**：双臂复位；**夹爪回初始位**：发送左右夹爪初始位置目标。
- 模型管理、四路视觉、状态、轨迹、历史记录和导出功能保留。加载页面或启动网页服务不会自动执行动作。

模型服务单独管理。输入有效、模型在线以及控制状态满足已有要求后，才可开始连续执行。本次迁移没有启动模型或执行真机动作。

## 目录

| 路径 | 内容 |
| --- | --- |
| `forcevla_web/` | FastAPI、前端、采集、推理、HOME、夹爪和闭环进程管理 |
| `robot_control/` | 动作转换、离线 IK/QP、MoveJ 闭环、反馈与停止处理、诊断工具 |
| `config/` | `local.json` 网页配置；`robot.json` 观察配置；`planning.json` 规划配置；`planner_loop.json` 闭环参数 |
| `native/` | 网页离线 IK 工作程序 |
| `tests/`、`tests/robot_control/` | 网页和控制回归测试 |
| `data/` | 原网页数据库、会话密钥、闭环/HOME/夹爪记录 |
| `data/integration-runs/` | 原控制集成运行记录和源码快照 |
| `artifacts/`、`artifacts/robot-control/` | 网页及控制验收材料 |
| `docs/web/`、`docs/control/` | 按来源整理的历史说明和验收文档 |

## 启动与日志

```bash
cd /home/user/forcevla-web
systemctl --user status forcevla-web.service
journalctl --user -u forcevla-web.service -n 80 --no-pager
```

访问 `http://172.30.1.208:8096/` 或 `http://marvin-thor.local:8096/`。允许访问的主机名在 `config/local.json` 配置。

需要手动启动时，先停止用户服务，再运行 `bash scripts/run.sh`，避免争用同一数据目录。

```bash
# 模拟输入及模拟策略，独立端口和数据目录
bash scripts/run.sh --demo --port 8097
# 控制工具统一入口；帮助命令不会发送动作
bash scripts/run_control.sh planner-loop --help
bash scripts/run_control.sh predict --help
# 纯模拟控制
bash scripts/run_control.sh control --mode simulate --output /tmp/forcevla-control-example
```

环境分别为 `.venv`（网页）、`.ros-venv`（ROS 采集）、`.control-venv`（控制）。模型仓库、权重和系统 SDK 保持各自原来的安装位置。新装环境可运行 `bash scripts/install.sh`；本次迁移复用原环境，未升级依赖。

## 验证

```bash
env -u PYTHONPATH -u VIRTUAL_ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests --ignore=tests/robot_control
bash scripts/run_control.sh test
npm run check
```

浏览器测试默认访问 8097 的模拟服务；可通过 `FORCEVLA_TEST_URL` 指定测试地址。历史验证结果描述当时的部署条件，当前迁移结果见 [迁移说明](MIGRATION_ZH.md)。

详细说明：[连续执行](docs/web/PLANNER_LOOP_ZH.md)、[HOME](docs/web/HOME_ZH.md)、[夹爪初始目标](docs/web/GRIPPER_INITIAL_TARGET_ZH.md)、[控制模块说明](docs/control/README.md)、[合并前网页说明](docs/web/README_BEFORE_MERGE_ZH.md)。
