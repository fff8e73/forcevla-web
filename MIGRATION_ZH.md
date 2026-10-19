# 2026-10-19 统一项目迁移

正式目录：`/home/user/forcevla-web`。正式服务：`forcevla-web.service`。服务于 2026-10-19 11:05（Asia/Shanghai）完成切换，仍使用端口 8096。

## 合并结果

原网页包 `shadow_web` 改为 `forcevla_web`；原 `forcevla_robot_integration` 的控制代码纳入 `robot_control` 包。网页从同一项目启动控制子进程，配置、依赖声明、测试和文档也纳入项目。控制模块使用显式包导入；原生程序的资格校验入口根据自身位置找到项目环境，不再依赖旧控制目录。

`config/shadow.json` 改为 `config/planning.json`，`run_shadow.py` 改为 `run_predict.py`，`shadow_checks.py` 改为 `prediction_checks.py`，`shadow_rpc.py` 改为 `model_rpc.py`。控制命令使用 `bash scripts/run_control.sh predict ...`。源码快照保留合并后的包结构。

网页标题/页脚、服务、包、前端项目和虚拟环境提示名使用 ForceVLA Web。普通推理 API 的 `mode` 标识由 `shadow` 改为 `prediction`；历史保存的标识不改写。会话 Cookie 使用 `forcevla_session`，仍接受旧 Cookie，以保持原浏览器的会话身份；任务文本也兼容读取旧 localStorage 键。

三个环境保持分离：`.venv` 用于网页，`.ros-venv` 用于采集，`.control-venv` 用于机器人控制。环境迁移修正启动脚本的绝对路径，没有升级已安装依赖。模型仓库、权重、ROS 和系统 SDK 继续使用原位置。

## 数据与兼容入口

迁移前已停止旧网页服务，并确认无运行中的推理、闭环、HOME、夹爪或控制验证任务。数据库和会话密钥先备份再迁移；启用新服务前校验内容未改变。

| 数据 | 校验结果 | 新位置 |
| --- | --- | --- |
| 网页 SQLite | 16 个会话、2378 条预测、4873 个事件，迁移前后文件 SHA256 相同 | `data/runs.sqlite3` |
| 浏览器会话密钥 | SHA256 相同 | `data/session.key` |
| 原控制历史 | 84 个运行目录、3192 个文件，逐文件 SHA256 相同 | `data/integration-runs/` |
| 网页验收材料 | 整目录迁移 | `artifacts/` |
| 控制验收材料 | 整目录迁移 | `artifacts/robot-control/` |
| 模拟网页历史 | 整目录迁移 | `data-demo/` |

两个旧顶层目录保留为兼容软链接，均指向统一项目。`robot_control` 下的 `config`、`runs`、`artifacts` 和 `.venv` 也是兼容入口。正式服务和当前代码使用新路径。历史报告中的旧绝对路径和当时的源码快照保留原样。

旧源码原件、旧服务配置、数据库备份、迁移操作日志和测试报告位于 `/home/user/forcevla-merge-20261019/`。其中 `original-web/` 和 `original-control/` 是迁移前源码；`runs-before.sqlite3` 与 `session-before.key` 是切换前备份。`cutover-journal.json` 记录逐项移动，`control-history-manifest.json` 保存控制历史校验值。

## 验证范围

网页测试、ROS 图像转换测试、控制回归、模拟控制和浏览器模拟流程已执行；浏览器模拟覆盖单次推理、停止、快照、历史、导出和手机布局。网页共验证 103 项（另有 1 项需 ROS 环境的跳过项），采集专项 6 项，控制共验证 216 项。网页全量复测中一项原生闭环模拟首次未达到预期轮数，单独复测通过；最终控制全量回归中一项执行时序测试触发 `native_deadline`，该模块单独复测 8 项全部通过。未放宽相关门槛。完整输出与 `validation-summary.json` 位于迁移目录。

重新编译了网页离线 IK 程序和原生控制程序。44 个 QP/模型资产校验全部匹配；纯模拟控制报告 `hardware_commands_sent=0`。原生资格校验测试在证据不存在时，于 ROS 初始化之前拒绝执行，并验证迁移后的模块可以从其他工作目录找到。

上线检查只读取页面和状态：确认新服务、实时输入、四路图像、历史记录和前端无脚本错误。本次没有发送真机动作，也没有加载或运行真实模型推理。切换前模型服务已经停止；切换后仍处于未加载状态，后续真实推理需要先准备模型。模型管理配置和注册项来自迁移前配置，未改成其他模型。

此前控制限值、停止行为和已存在的执行约束未因迁移放宽。原生执行的资格记录仍按源码和模型身份检查，旧记录不会被迁移自动认定为有效。

## 恢复原部署

若需要恢复，应先确认无运行中的控制任务并停止 `forcevla-web.service`，按照 `cutover-journal.json` 反向迁移：移除旧路径软链接、恢复两个 `original-*` 源码目录，将控制历史/验收材料/控制环境分别迁回，再迁回网页数据/环境/验收材料，恢复环境启动脚本的旧路径，最后恢复旧服务配置并启用。恢复时保留切换后产生的数据；不要用切换前数据库备份覆盖新记录。迁移脚本自身在切换失败时会自动执行反向恢复；成功上线后不应重复运行该脚本。
