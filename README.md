# ForceVLA Web

统一的模型推理与机器人控制网页，包含四路视觉输入、状态观察、单次推理、连续真机闭环、HOME、夹爪初始位置控制、历史记录和导出。

网页代码位于 `forcevla_web/`，机器人控制位于 `robot_control/`。控制任务通过独立进程执行；网页、ROS 采集和控制分别使用自己的 Python 环境。

## 获取代码

```bash
git clone --recurse-submodules https://github.com/fff8e73/forcevla-web.git
cd forcevla-web
cp config/local.example.json config/local.json
```

已有克隆可运行 `git submodule update --init --recursive` 补齐固定版本的 DAQP 依赖。`robot_control/vendor/daqp` 固定到提交 `653e22316c1d54ee7c71500f45f3ff7afaadad07`；原生构建脚本核对该版本及其工作区状态。

部署面向本机 ROS 2 Jazzy 和 Apex/Marvin SDK 环境。先按目标机器修改 `config/local.json` 中的仓库、模型、数据、原生程序及允许访问的主机名等路径；`config/robot.json`、`config/planning.json` 也包含对应机器的配置与资产校验值，需要核对实际安装。模型权重、系统 SDK、虚拟环境、运行记录和编译结果未包含在仓库中。

## 文档

- [部署、页面功能与测试说明](README_ZH.md)
- [项目合并与历史数据迁移说明](MIGRATION_ZH.md)
- [网页功能与验收记录](docs/web/README_BEFORE_MERGE_ZH.md)
- [机器人控制说明](docs/control/README.md)

`config/local.json`、运行数据、会话密钥和本机环境由 `.gitignore` 排除；仓库提供 `config/local.example.json` 作为配置起点。文档中引用的历史运行材料保存在部署机器上，部分链接需要在该机器访问。

## 常用入口

```bash
bash scripts/run.sh --demo --port 8097
bash scripts/run_control.sh planner-loop --help
```

安装依赖及真实部署操作见部署说明。页面的“开始连续”、HOME 和夹爪控制会操作真实设备；模拟模式使用模拟输入与策略。
