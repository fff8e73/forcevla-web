const $ = (id) => document.getElementById(id);
let api,
  data = null,
  busy = false,
  inferenceBusy = false;
const active = new Set([
  "CHECKING",
  "SELECTING_NONE",
  "CLAIMING",
  "OBSERVING",
  "HOLDING",
  "RELEASING",
  "ABORTING",
]);
const review = new Set(["HELD_REVIEW", "UNCERTAIN", "INTERRUPTED_REVIEW"]);
const names = {
  CHECKING: "检查静止条件",
  SELECTING_NONE: "切换至 None",
  CLAIMING: "继承目标接管",
  OBSERVING: "观察真实反馈",
  HOLDING: "确认保持",
  RELEASING: "释放本会话",
  ABORTING: "正在中止",
  COMPLETED_STATIONARY: "静止链路验证完成",
  BLOCKED: "前置检查未通过",
  ABORTED_UNOWNED: "已中止 · 未占用",
  HELD_REVIEW: "已保持 · 等待明确释放",
  UNCERTAIN: "状态未确认 · 请检查设备",
  INTERRUPTED_REVIEW: "服务重启 · 等待检查",
  RELEASED: "本会话已释放",
};
export function renderControlValidation(value, inferenceActive) {
  if (typeof inferenceActive === "boolean") inferenceBusy = inferenceActive;
  data = value;
  const run = data?.run,
    running = active.has(run?.state),
    pending = review.has(run?.state);
  $("cv-state").textContent = run ? names[run.state] || run.state : "未启动";
  $("cv-state").className =
    `tag ${pending || run?.state === "BLOCKED" ? "bad" : ""}`;
  $("cv-reason").textContent =
    run?.reason || "先做只读检查。页面加载、推理和刷新都不会启动真实测试。";
  $("cv-check").disabled = busy || running || !data?.enabled;
  const ready = data?.last_check?.ready === true;
  const acknowledged = $("cv-safety").checked;
  for (const id of ["cv-cycle", "cv-lease"])
    $(id).disabled =
      busy ||
      !data?.enabled ||
      running ||
      pending ||
      inferenceBusy ||
      !ready ||
      !acknowledged;
  $("cv-abort").disabled = busy || !running || !run?.is_owner;
  $("cv-release").disabled =
    busy || !pending || !run?.is_owner || !acknowledged;
  $("cv-export").disabled = !run;
  $("cv-safety").disabled = running;
  $("cv-moving").disabled = true;
  const check = data?.last_check;
  $("cv-precheck").textContent = check
    ? check.ready
      ? "只读条件满足：启动时仍会重新核验。"
      : check.reasons.join("；")
    : "尚未进行只读检查";
  const m = check?.measurements;
  $("cv-feedback").textContent = m
    ? `关节最大速度 ${format(m.joint_max_speed_rad_s)} rad/s · 左夹爪 ${format(m.left_gripper?.velocity_rad_s)} rad/s · 右夹爪 ${format(m.right_gripper?.velocity_rad_s)} rad/s\nRobot Mode ${check.robot_mode} · Input ${check.input_mode} · 原夹爪目标 ${JSON.stringify(check.target_gripper)}
网页夹爪速度门槛 ${format(check.web_gripper_speed_limit_rad_s)} rad/s；驱动接管检查仍为 0.0200 rad/s`
    : "等待新鲜关节与电机速度";
  const events = run?.events || [];
  $("cv-events").replaceChildren(
    ...events.slice(-30).map((event) => {
      const row = document.createElement("div");
      row.className = "cv-event";
      const time = document.createElement("time");
      time.textContent = new Date(event.time_unix * 1000).toLocaleTimeString();
      const text = document.createElement("span");
      text.textContent = event.message;
      row.append(time, text);
      return row;
    }),
  );
  $("cv-owner").textContent =
    run && !run.is_owner && (running || pending)
      ? "其他浏览器持有本验证会话，仅可查看。"
      : "";
  $("cv-result").textContent = run
    ? `新运动目标：${run.target_commands_sent || 0} 条 · 接口操作：${run.post_count || 0} 次 · 运动中停止：未验证`
    : "本流程不发送新关节/TCP/夹爪目标，不改变 Robot Mode。";
}
function format(value) {
  return Number.isFinite(value) ? value.toFixed(4) : "—";
}
function error(message) {
  $("cv-message").textContent = message;
}
export function initControlValidation(request) {
  api = request;
  async function perform(path, body) {
    busy = true;
    error("");
    renderControlValidation(data);
    try {
      return await api("/control-validation/" + path, {
        method: "POST",
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      error(e.message);
    } finally {
      busy = false;
      const fresh = await api("/control-validation/status").catch(() => null);
      if (fresh) renderControlValidation(fresh);
    }
  }
  $("cv-check").onclick = () => perform("check");
  for (const [id, kind, label] of [
    ["cv-cycle", "stationary_cycle", "静止接管、保持并释放"],
    ["cv-lease", "lease_expiry", "租约超时保持并释放"],
  ]) {
    $(id).onclick = async () => {
      if (
        !confirm(
          `确认在真实设备上执行“${label}”？\n输入来源将切为 None，结束后保持 None。不会发送新运动目标；这不验证运动中停止。`,
        )
      )
        return;
      await perform("start", {
        kind,
        command_id: Array.from(
          crypto.getRandomValues(new Uint8Array(16)),
          (b) => b.toString(16).padStart(2, "0"),
        ).join(""),
        safety_ack: $("cv-safety").checked,
      });
    };
  }
  $("cv-abort").onclick = () => perform("abort");
  $("cv-release").onclick = async () => {
    if (!confirm("确认释放本次验证留下的夹爪占用？Input 必须仍为 None。"))
      return;
    await perform("release", { safety_ack: $("cv-safety").checked });
  };
  $("cv-export").onclick = () =>
    location.assign("/api/v1/control-validation/report");
  $("cv-safety").onchange = () => renderControlValidation(data);
}
