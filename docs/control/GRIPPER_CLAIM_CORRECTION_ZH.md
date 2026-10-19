> 历史说明：本文记录合并前的功能与验收过程；当前目录、命令及部署状态以 [统一项目说明](../../README_ZH.md) 为准。历史报告内容保留原有含义。

# 夹爪接管判断勘误

2026-10-14：此前“actual为负，所以驱动claim必然拒绝”的结论错误，应撤回。它漏读了WebGripper.command的外层处理：claim时先检查电机速度绝对值≤0.02 rad/s、实际位置与原目标差≤0.05 rad，然后把现有q1/q2目标归一化作为seed传给Gate.command。内层0～1检查作用于这个seed，不直接作用于实际负反馈。

目标0、实际−0.034 rad且静止时，健康边界和跟踪容差均允许，seed为0。使用已安装源码的无硬件模拟测试验证此条件下claim成功，继承目标0；速度超限或跟踪偏差超限则拒绝。没有实例化真实WebGripper、初始化ROS/CAN或发送claim。

已修正hardware_backend.gripper_claim_readiness及测试。HTTP状态缺少电机速度，因此没有该独立证据时driver_claim_preconditions_passed为null（未知），不再因为actual略负而直接判失败。报告保留actual，不裁剪、不改标定。

最终回归69项测试通过：[日志](../../data/integration-runs/claim-wrapper-correction-tests.log)。[当前只读检查](../../data/integration-runs/20261014-claim-wrapper-corrected.json)。历史JSON保持原样，但其中gripper_actual_outside_claim_range的接管阻塞结论无效，应以本勘误为准。

实际反馈容差与目标范围本来是两类检查，不能说二者不同就代表驱动bug。模型预测的负目标仍属于独立问题：模型输出不合法不等于实际反馈不健康；本勘误不放行越界预测、不解锁真实硬件后端。
