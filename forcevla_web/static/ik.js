import { joints } from "./plots.js";
const $ = (id) => document.getElementById(id);
const labels = {
  seed_joint_limit: "起点关节超限",
  seed_tcp_mismatch: "起点FK与TCP反馈不一致",
  qp_failed: "QP求解失败",
  ik_not_converged: "未找到满足误差阈值的解",
  joint_limit: "关节超限",
  joint_jump: "关节跳变",
  joint_velocity: "关节速度超限",
  gripper_range: "夹爪目标超范围",
  configured_collision_clearance: "已配置几何体间距不足",
  collision_sampling_budget: "插值检查超出采样预算",
  solve_time_budget: "求解时间预算耗尽",
};
export function renderIK(prediction, step, view, globalStatus) {
  const raw = prediction?.offline_ik,
    report = raw && typeof raw === "object" ? raw : null;
  const state = report?.status;
  const texts = {
    pending: "正在离线检查…",
    interrupted: "检查中断",
    not_requested: "预测已丢弃，未检查",
    complete: report?.passed ? "离线检查通过" : "离线检查未通过",
    budget_exceeded: "检查未完成：时间预算",
    seed_unavailable: "关节快照不可用",
    invalid_seed: "起点检查未通过",
    unavailable: "离线求解器不可用",
    error: "检查发生错误",
    superseded: "已跳过旧的待检查预测",
  };
  $("ik-summary").textContent = prediction
    ? texts[state] || "该历史预测尚未检查"
    : "等待预测";
  $("ik-summary").className = report?.passed
    ? "good"
    : state === "pending"
      ? "subtle"
      : "bad";
  $("ik-service").textContent = globalStatus?.available
    ? "离线求解器在线"
    : globalStatus?.enabled
      ? "离线求解器不可用"
      : "离线检查未启用";
  $("ik-service-note").textContent =
    globalStatus?.error || "完整 m6_696 · 只计算，不下发动作";
  const row = report?.steps?.[step];
  const reasons = row?.reasons?.length ? row.reasons : report?.reasons || [];
  $("ik-reasons").textContent =
    report?.error ||
    reasons.map((r) => labels[r] || r).join("；") ||
    (report?.passed ? "全部目标满足当前离线检查门槛" : "");
  $("ik-step-info").textContent = row
    ? `第 ${step + 1} 步 · ${row.converged ? "IK收敛" : "IK未收敛"} · 迭代 ${row.iterations} 次 · 已检查 ${report.checked_steps}/${report.total_steps} 步`
    : report?.checked_steps != null
      ? `已检查 ${report.checked_steps}/${report.total_steps} 步；当前步无结果`
      : "每份检查绑定原始观察和关节快照";
  $("ik-residual").textContent = row
    ? `位置误差 L/R ${(row.residual[0].position_m * 1000).toFixed(2)} / ${(row.residual[1].position_m * 1000).toFixed(2)} mm；旋转误差 ${row.residual.map((r) => ((r.orientation_rad * 180) / Math.PI).toFixed(2)).join(" / ")}°`
    : "—";
  $("ik-collision").textContent = row
    ? `所检区间最小间距 ${(row.closest.distance_m * 1000).toFixed(1)} mm · ${row.closest.pair.join(" ↔ ")}；插值检查 ${row.interpolation_samples}/${row.interpolation_samples_required} 点`
    : "—";
  $("ik-jumps").textContent = row
    ? `最大关节步变 ${((row.max_joint_step_rad * 180) / Math.PI).toFixed(2)}° · 最小限位余量 ${((row.joint_margin_rad * 180) / Math.PI).toFixed(2)}°`
    : "—";
  $("ik-context").textContent = report?.seed_context
    ? `起点快照 ${report.seed_context.snapshot_id} · 求解 ${(report.elapsed_s * 1000).toFixed(1)} ms${report.historical ? " · 历史检查" : ""}`
    : "旧记录若没有关节反馈，将不会使用当前关节角补算";
  $("ik-joints").replaceChildren(
    ...Array.from({ length: 7 }, (_, i) => {
      const tr = document.createElement("tr");
      for (const value of [
        `J${i + 1}`,
        row ? `${((row.q[i] * 180) / Math.PI).toFixed(2)}°` : "—",
        row ? `${((row.q[i + 7] * 180) / Math.PI).toFixed(2)}°` : "—",
      ]) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      }
      return tr;
    }),
  );
  joints(
    $("joint-pose"),
    report?.seed?.skeleton,
    row?.skeleton,
    view,
    row && !row.passed,
  );
  $("view-checked").disabled = !globalStatus?.latest_checked_prediction_id;
}
