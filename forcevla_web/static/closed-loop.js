const $ = (id) => document.getElementById(id);
const terminal = new Set([
  "COMPLETED_LIMIT",
  "CANCELLED",
  "TIMEOUT",
  "FAILED",
  "TASK_SUCCESS",
  "STOP_UNCONFIRMED",
  "INTERRUPTED_REVIEW",
]);
const names = {
  IDLE: "未启动",
  STARTING: "启动中",
  PRECHECK: "预检",
  WARMING: "模型预热",
  ACQUIRE: "接管",
  HOLDING: "保持已确认",
  OBSERVE: "等待新观察",
  INFER: "推理中",
  PREPARE: "准备动作",
  EXECUTE: "执行中",
  SETTLING: "等待保持确认",
  PAUSING: "正在暂停并保持",
  PAUSED: "已暂停，等待连接恢复",
  RESUMING: "连接恢复，重新观察",
  STOPPING: "停止请求处理中",
  COMPLETED_LIMIT: "已完成指定轮数",
  CANCELLED: "已取消",
  TIMEOUT: "已超时",
  FAILED: "运行失败",
  TASK_SUCCESS: "任务成功",
  STOP_UNCONFIRMED: "停止未确认",
  INTERRUPTED_REVIEW: "中断待复核",
};
let api,
  current = null,
  overall = null,
  busy = false,
  historical = null,
  historyVersion = 0,
  defaultsApplied = false,
  connected = false,
  inferencePending = false;
const reasonNames = {
  browser_heartbeat_timeout: "浏览器心跳应答超时",
  browser_disconnected: "浏览器心跳超时（旧版记录）",
  invalid_browser_heartbeat: "浏览器心跳数据异常，已停止",
  parent_disconnected: "网页服务监督中断，已停止",
};
const fmt = (v) => (Number.isFinite(v) ? `${Math.round(v * 1000)} ms` : "—");
function renderRun(run) {
  $("cl-state").textContent = names[run?.state] || run?.state || "未启动";
  $("cl-progress").textContent = run
    ? `${run.completed_cycles || 0} / ${run.config?.max_cycles || "—"} 轮完成`
    : "—";
  $("cl-age").textContent = Number.isFinite(run?.prediction_age_s)
    ? `${run.prediction_age_s.toFixed(2)} s${terminal.has(run.state) ? "（结束时）" : ""}`
    : "—";
  $("cl-owner").textContent = run
    ? `${run.is_owner ? "本浏览器会话" : "其他浏览器会话"}${run.execution?.session ? ` · 控制会话 ${run.execution.session.slice(0, 8)}` : ""} · 提交计数与实际反馈分别记录`
    : "";
  const health =
    run?.connection_diagnostics?.connection_health ||
    (!historical ? overall?.connection_health : null);
  if (health?.protocol === "server_challenge_v1") {
    const visibility =
      health.visibility_state === "hidden"
        ? "页面在后台"
        : health.visibility_state === "visible"
          ? "页面可见"
          : "页面状态未知";
    $("cl-owner").textContent +=
      ` · 心跳往返 ${fmt(Number.isFinite(health.rtt_ms) ? health.rtt_ms / 1000 : null)} · ${visibility}`;
  }
  const e = run?.execution;
  const paused = ["PAUSING", "PAUSED", "RESUMING"].includes(run?.state);
  $("cl-executed").textContent = run
    ? `${e?.executed_model_steps || 0} / ${run.config?.execution_horizon || "—"} 步${run.engine === "movej" ? "已到位" : ` · ${e?.current_segment_ticks || 0} 个控制周期`}`
    : "—";
  $("cl-timing").textContent = `${fmt(run?.rpc_s)} / ${fmt(run?.prepare_s)}`;
  $("cl-hold").textContent = run?.stop
    ? run.stop.confirmed
      ? paused ? "暂停保持已确认" : "最终保持已确认"
      : "停止未确认"
    : e?.phase === "HOLDING"
      ? "保持已确认"
      : e?.phase === "SETTLING"
        ? "等待反馈确认"
        : "—";
  const mode = run?.mode === "simulate" ? "模拟数据" : run ? "真机" : "";
  $("cl-result").textContent = run
    ? `${historical ? "历史 · " : ""}${mode} · ${names[run.state] || run.state}${run.reason ? ` · ${reasonNames[run.reason === "operator_cancelled" && run.connection_diagnostics?.reason === "browser_heartbeat_timeout" ? "browser_heartbeat_timeout" : run.reason] || run.reason}` : ""}`
    : "尚无闭环记录";
  if (run?.state === "PAUSED") {
    const stable = Math.min(3, run.recovery?.stable_s || 0).toFixed(1);
    $("cl-result").textContent += ` · 同一会话心跳稳定 ${stable}/3 秒后自动重新观察和推理`;
  }
  $("cl-cycle-log").textContent =
    run?.history
      ?.map(
        (c) =>
          `第${c.cycle}轮 · 新观察 ${c.observation_id} · 执行${c.executed_steps}步 · 保持${c.hold_confirmed ? "已确认" : "未确认"} · RPC ${fmt(c.rpc_s)}`,
      )
      .join("\n") || "每轮观察、执行和保持结果将在这里显示";
  $("cl-export").disabled = !run?.id || !terminal.has(run.state);
}
function locks() {
  const mode = $("cl-mode").value;
  const steps = Number($("cl-k").value);
  const cycles = Number($("cl-cycles").value);
  const validParameters =
    ["cl-k", "cl-cycles", "cl-duration"].every(
      (id) => $(id).value !== "" && $(id).checkValidity(),
    ) &&
    Number.isInteger(steps) &&
    Number.isInteger(cycles);
  const stepLabel =
    Number.isInteger(steps) && steps >= 1 && steps <= 50 ? steps : "K";
  $("continuous").title =
    mode === "planner"
      ? `执行真机动作：最多${Number.isInteger(cycles) && cycles > 0 ? cycles : "—"}轮，每轮${stepLabel}步`
      : "使用真机闭环参数启动连续执行";
  const predictionBusy = [
    "WARMING",
    "RUNNING",
    "WAITING_INPUT",
    "STOPPING",
  ].includes(overall?.run?.state);
  const available =
    current?.enabled &&
    (mode === "simulate" ||
      (mode === "planner"
        ? current?.planner_available
        : current?.live_available));
  $("cl-start").textContent =
    mode === "simulate"
      ? "开始模拟闭环"
      : mode === "planner"
        ? `开始${stepLabel}步真机闭环`
        : "开始原生节拍验证";
  $("cl-start").hidden = mode === "planner";
  $("cl-start").disabled =
    !connected ||
    inferencePending ||
    busy ||
    !available ||
    !validParameters ||
    current?.blocked ||
    predictionBusy ||
    overall?.gripper_restart?.blocked ||
    overall?.home?.blocked ||
    overall?.control_validation?.blocked ||
    (!(overall?.model?.healthy && overall?.input?.ready) &&
      mode === "planner") ||
    !$("prompt").value.trim();
  $("cl-stop").disabled =
    !connected || busy || !current?.run?.is_owner || !current?.blocked;
  for (const id of ["cl-mode", "cl-k", "cl-cycles", "cl-duration"])
    $(id).disabled = busy || current?.blocked;
  $("continuous").disabled =
    !connected ||
    inferencePending ||
    busy ||
    !current?.planner_available ||
    (mode === "planner" && !validParameters) ||
    current?.blocked ||
    predictionBusy ||
    overall?.gripper_restart?.blocked ||
    overall?.home?.blocked ||
    overall?.control_validation?.blocked ||
    !overall?.model?.healthy ||
    !overall?.input?.ready ||
    Boolean(overall?.storage_error) ||
    !$("prompt").value.trim();
  $("execute20-stop").disabled = $("cl-stop").disabled;
  $("cl-mode-note").textContent = !current?.enabled
    ? "分段闭环尚未启用"
    : mode === "simulate"
      ? "模拟验证使用合成观察和模拟执行器；上方实时相机不代表模拟执行画面。"
      : mode === "planner" && current?.planner_available
        ? `每轮读取新观察、推理并执行前${stepLabel}步，逐点确认到位后继续。可随时停止；运行节拍由现有规划器决定。`
        : current?.live_available
          ? "按已配置的执行范围运行。"
          : `真机执行尚不可用：${(current?.live_blockers || []).join("；")}`;
  $("cl-profile").textContent =
    mode === "planner"
      ? `最多${Number.isInteger(cycles) && cycles > 0 ? cycles : "—"}轮 · 每轮${stepLabel}步 · 每次模型返回50步，执行所选前缀后重新观察 · 心跳超时先暂停保持，同一会话恢复稳定3秒后重新推理 · 达到轮数或时长上限后停止`
      : current?.profile
        ? `每次预测50步 · 目标间隔1/30秒 · ${current.profile.browser_disconnect_timeout_s}秒未收到本会话心跳应答后停止 · 达到轮数不代表插接成功`
        : "";
}
export function setClosedLoopConnection(value, pending = false) {
  connected = value;
  inferencePending = pending;
  locks();
}
export function renderClosedLoop(value, status) {
  $("closed-loop-panel").hidden = !value?.enabled;
  current = value;
  overall = status;
  if (!defaultsApplied && value) {
    defaultsApplied = true;
    if (!value.blocked) {
      if (value.planner_available) setPlannerDefaults();
      else {
        $("cl-mode").value = "simulate";
        $("cl-k").max = "3";
        $("cl-k").value = value.profile?.execution_horizon || 3;
        $("cl-cycles").value = value.profile?.max_cycles || 3;
        $("cl-duration").value = value.profile?.max_task_duration_s || 30;
      }
    }
  }
  if (value?.blocked && value.run?.config) {
    $("cl-mode").value =
      value.run.engine === "movej" ? "planner" : value.run.mode;
    $("cl-k").max = value.run.engine === "movej" ? "50" : "3";
    $("cl-k").value = value.run.config.execution_horizon;
    $("cl-cycles").value = value.run.config.max_cycles;
    $("cl-duration").value = value.run.config.max_task_duration_s;
  }
  if (!historical) renderRun(value?.run);
  locks();
}
function setPlannerDefaults() {
  $("cl-mode").value = "planner";
  $("cl-k").max = "50";
  $("cl-k").value = current?.planner_profile?.execution_horizon || 20;
  $("cl-cycles").value = current?.planner_profile?.max_cycles || 50;
  $("cl-duration").value = current?.planner_profile?.max_task_duration_s || 600;
}
async function refresh() {
  const selected = $("cl-history").value;
  const rows = await api("/closed-loop/runs");
  $("cl-history").replaceChildren(
    new Option("当前会话", ""),
    ...rows.map(
      (r) =>
        new Option(
          `${new Date(r.created_unix * 1000).toLocaleString()} · ${r.mode === "simulate" ? "模拟" : "真机"} · ${names[r.state] || r.state}`,
          r.id,
        ),
    ),
  );
  $("cl-history").value = selected;
}
export function initClosedLoop(client) {
  api = client;
  $("cl-start").onclick = async () => {
    if ($("cl-start").disabled) return;
    busy = true;
    locks();
    $("cl-message").textContent = "";
    try {
      for (const id of ["cl-k", "cl-cycles", "cl-duration"]) {
        if (!$(id).value || !$(id).reportValidity())
          throw Error("请填写有效的轮数、步数和时长");
      }
      const seedText = $("seed").value;
      const options = {
        mode: $("cl-mode").value === "planner" ? "live" : $("cl-mode").value,
        engine: $("cl-mode").value === "planner" ? "movej" : "native",
        prompt: $("prompt").value.trim(),
        seed: seedText === "" ? 0 : Number(seedText),
        execution_horizon: Number($("cl-k").value),
        max_cycles: Number($("cl-cycles").value),
        max_task_duration_s: Number($("cl-duration").value),
        command_id: `loop-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      };
      const run = await api("/closed-loop/runs", {
        method: "POST",
        body: JSON.stringify(options),
      });
      historical = null;
      ++historyVersion;
      $("cl-history").value = "";
      current = { ...current, run, blocked: true };
      renderRun(run);
      $("cl-message").textContent =
        options.engine === "movej"
          ? `已启动连续真机闭环：最多${options.max_cycles}轮，每轮${options.execution_horizon}步`
          : "已启动有限轮次闭环";
      await refresh();
    } catch (e) {
      $("cl-message").textContent = e.message;
    } finally {
      busy = false;
      locks();
    }
  };
  $("cl-stop").onclick = async () => {
    busy = true;
    locks();
    try {
      await api(`/closed-loop/runs/${current.run.id}/stop`, { method: "POST" });
      $("cl-message").textContent = "停止请求已发送，等待实际保持确认";
    } catch (e) {
      $("cl-message").textContent = e.message;
    } finally {
      busy = false;
      locks();
    }
  };
  $("cl-mode").onchange = () => {
    if ($("cl-mode").value === "planner") setPlannerDefaults();
    else {
      $("cl-k").max = "3";
      $("cl-k").value = "3";
      $("cl-cycles").value = "3";
      $("cl-duration").value = "30";
    }
    locks();
  };
  for (const id of ["cl-k", "cl-cycles", "cl-duration"]) $(id).oninput = locks;
  $("continuous").onclick = () => {
    if ($("continuous").disabled) return;
    if ($("cl-mode").value !== "planner") setPlannerDefaults();
    locks();
    $("closed-loop-panel").scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
    $("cl-start").onclick();
  };
  $("execute20-stop").onclick = () => $("cl-stop").onclick();
  $("prompt").addEventListener("input", locks);
  $("cl-refresh").onclick = () =>
    refresh().catch((e) => {
      $("cl-message").textContent = e.message;
    });
  $("cl-history").onchange = async () => {
    const version = ++historyVersion;
    const id = $("cl-history").value;
    historical = id || null;
    if (!id) {
      renderRun(current?.run);
      return;
    }
    try {
      const run = await api(`/closed-loop/runs/${id}`);
      if (version === historyVersion) {
        historical = run;
        renderRun(run);
      }
    } catch (e) {
      $("cl-message").textContent = e.message;
    }
  };
  $("cl-export").onclick = () => {
    const run =
      typeof historical === "object" && historical ? historical : current?.run;
    if (run?.id) location.assign(`/api/v1/closed-loop/runs/${run.id}/export`);
  };
  if (current?.enabled)
    refresh().catch((e) => {
      $("cl-message").textContent = e.message;
    });
  locks();
}
