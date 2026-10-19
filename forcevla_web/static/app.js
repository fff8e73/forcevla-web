import {
  initClosedLoop,
  renderClosedLoop,
  setClosedLoopConnection,
} from "./closed-loop.js";
import {
  initControlValidation,
  renderControlValidation,
} from "./control-validation.js";
import { renderIK } from "./ik.js";
import { trajectory, lines } from "./plots.js";
const $ = (id) => document.getElementById(id);
const cameras = [
  ["image", "左眼 · HEAD L"],
  ["right_image", "右眼 · HEAD R"],
  ["wrist_image", "左腕 · WRIST L"],
  ["right_wrist_image", "右腕 · WRIST R"],
];
let csrf = "",
  socket,
  status = null,
  prediction = null,
  predictionID = null,
  loadingPredictionID = null,
  view = "live",
  busy = false,
  homeSubmitting = false,
  homePending = null,
  gripperRestartSubmitting = false,
  gripperRestartPending = null,
  historyRun = "",
  historyPrediction = "",
  fetchVersion = 0,
  historyVersion = 0,
  connected = false;
let modelManager = null;
const latencies = [],
  seenPredictions = new Set();
for (const [key, title] of cameras) {
  const el = document.createElement("div");
  el.className = "camera";
  el.id = `camera-${key}`;
  const image = document.createElement("img");
  image.alt = title;
  image.hidden = true;
  const placeholder = document.createElement("div");
  placeholder.className = "camera-placeholder";
  placeholder.textContent = "等待图像";
  const label = document.createElement("span");
  label.className = "camera-label";
  label.textContent = title;
  const state = document.createElement("span");
  state.className = "camera-state";
  state.textContent = "WAITING";
  image.onload = () => {
    image.hidden = false;
    placeholder.hidden = true;
  };
  image.onerror = () => {
    image.hidden = true;
    placeholder.hidden = false;
  };
  el.append(image, placeholder, label, state);
  $("camera-grid").append(el);
}
const fmt = (v, n = 3) => (Number.isFinite(v) ? v.toFixed(n) : "—");
async function api(path, options = {}) {
  const response = await fetch("/api/v1" + path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": csrf,
      ...options.headers,
    },
  });
  if (!response.ok) {
    let error;
    try {
      error = (await response.json()).detail;
    } catch {
      error = response.statusText;
    }
    throw Error(typeof error === "string" ? error : JSON.stringify(error));
  }
  return response.json();
}
function message(text) {
  $("operation-message").textContent = text;
}
function renderModelManager(value) {
  modelManager = value;
  const select = $("model-select");
  const current = select.value;
  const optionSignature = value.available.map((item) => `${item.id}:${item.name}`).join("|");
  if (select.dataset.signature !== optionSignature) {
    select.replaceChildren(
      ...value.available.map((item) => new Option(item.name, item.id)),
    );
    select.dataset.signature = optionSignature;
  }
  if (value.selected?.id) select.value = value.selected.id;
  else if (current && [...select.options].some((option) => option.value === current)) select.value = current;
  else if (select.options.length) select.selectedIndex = 0;
  const loading = ["STARTING", "STOPPING"].includes(value.state);
  $("model-manager-state").textContent = value.state;
  $("model-manager-state").className = `tag ${value.state === "RUNNING" ? "good" : value.state === "ERROR" ? "bad" : ""}`;
  $("model-manager-detail").textContent = value.error || (value.selected
    ? `${value.selected.name} · ${value.selected.endpoint || "等待 endpoint"}${value.metadata ? ` · schema ${JSON.stringify(value.metadata.metadata || {})}` : ""}`
    : "选择已登记的 PhyAI 部署包。加载过程会启动独立模型服务并检查 RPC metadata。");
  $("model-service-name").textContent = value.selected?.name || (value.available.find((item) => item.id === select.value)?.name || "未选择部署包");
  const progress = Math.max(0, Math.min(100, Number(value.progress || 0)));
  $("model-progress-bar").style.width = `${progress}%`;
  $("model-progress-stage").textContent = value.stage || "等待加载";
  $("model-progress-value").textContent = `${progress}% · ${Math.round(Number(value.elapsed_s || 0))}s`;
  $("model-log").textContent = value.log_tail || "等待模型日志…";
  $("model-log").scrollTop = $("model-log").scrollHeight;
  $("model-load").disabled = loading || value.state === "RUNNING" || !select.value;
  $("model-stop").disabled = loading || value.state !== "RUNNING";
}
async function refreshModels() {
  try { renderModelManager(await api("/models")); } catch (error) { message(`模型管理器：${error.message}`); }
}
setInterval(refreshModels, 1000);
function homeState() {
  return homePending || status?.home;
}
function gripperRestartState() {
  return gripperRestartPending || status?.gripper_restart;
}
function lockButtons() {
  renderControlValidation(
    status?.control_validation,
    ["WARMING", "RUNNING", "WAITING_INPUT", "STOPPING"].includes(
      status?.run?.state,
    ) ||
      Boolean(status?.closed_loop?.blocked) ||
      Boolean(homeState()?.blocked) ||
      Boolean(gripperRestartState()?.blocked),
  );
  const active = ["WARMING", "RUNNING", "WAITING_INPUT", "STOPPING"].includes(
    status?.run?.state,
  );
  const can =
    connected &&
    !busy &&
    !gripperRestartSubmitting &&
    !gripperRestartState()?.blocked &&
    !homeSubmitting &&
    !homeState()?.blocked &&
    status?.model.healthy &&
    status?.input.ready &&
    !active &&
    !status?.storage_error &&
    !status?.closed_loop?.blocked &&
    !status?.control_validation?.blocked &&
    $("prompt").value.trim();
  $("single").disabled = !can;
  setClosedLoopConnection(
    connected,
    busy ||
      homeSubmitting ||
      Boolean(homeState()?.blocked) ||
      gripperRestartSubmitting ||
      Boolean(gripperRestartState()?.blocked),
  );
  $("stop").disabled =
    !connected ||
    busy ||
    !active ||
    !status?.run?.is_owner ||
    status?.run?.state === "STOPPING";
  $("home").disabled =
    !connected ||
    gripperRestartSubmitting ||
    Boolean(gripperRestartState()?.blocked) ||
    homeSubmitting ||
    busy ||
    !homeState()?.enabled ||
    active ||
    Boolean(status?.closed_loop?.blocked) ||
    Boolean(status?.control_validation?.blocked) ||
    (homeState()?.blocked &&
      !["UNCERTAIN", "INTERRUPTED_REVIEW"].includes(homeState()?.run?.state));
  $("restart-gripper").disabled =
    !connected ||
    busy ||
    gripperRestartSubmitting ||
    homeSubmitting ||
    Boolean(homeState()?.blocked) ||
    !gripperRestartState()?.enabled ||
    active ||
    Boolean(status?.closed_loop?.blocked) ||
    Boolean(status?.control_validation?.blocked) ||
    (gripperRestartState()?.blocked &&
      !["UNCERTAIN", "INTERRUPTED_REVIEW"].includes(
        gripperRestartState()?.run?.state,
      ));
  $("prompt").disabled =
    active ||
    Boolean(status?.closed_loop?.blocked) ||
    Boolean(homeState()?.blocked) ||
    Boolean(gripperRestartState()?.blocked);
  $("seed").disabled =
    active ||
    Boolean(status?.closed_loop?.blocked) ||
    Boolean(homeState()?.blocked) ||
    Boolean(gripperRestartState()?.blocked);
  $("view-snapshot").disabled = !prediction;
  $("export").disabled = !(historyRun || status?.run?.id);
}
function feedback(state) {
  for (const [name, offset] of [
    ["left", 0],
    ["right", 13],
  ]) {
    const rows = [
      [
        "X / Y / Z",
        state
          .slice(offset, offset + 3)
          .map((v) => fmt(v))
          .join(" / "),
      ],
      [
        "R / P / Y",
        state
          .slice(offset + 3, offset + 6)
          .map((v) => fmt(v, 2))
          .join(" / "),
      ],
      ["夹爪", fmt(state[offset + 6])],
      [
        "Fx / Fy / Fz",
        state
          .slice(offset + 7, offset + 10)
          .map((v) => fmt(v, 1))
          .join(" / "),
      ],
      [
        "Tx / Ty / Tz",
        state
          .slice(offset + 10, offset + 13)
          .map((v) => fmt(v, 2))
          .join(" / "),
      ],
    ];
    const root = $(`${name}-feedback`);
    root.replaceChildren(
      ...rows.map(([name, value]) => {
        const row = document.createElement("div"),
          dt = document.createElement("dt"),
          dd = document.createElement("dd");
        dt.textContent = name;
        dd.textContent = value;
        row.append(dt, dd);
        return row;
      }),
    );
  }
}
function draw() {
  renderIK(
    prediction,
    Number($("step").value),
    $("projection").value,
    status?.ik,
  );
  trajectory(
    $("trajectory"),
    prediction,
    status?.state,
    Number($("step").value),
    $("projection").value,
  );
  const a = prediction?.actions || [];
  const grip = a.flatMap((x) => [x[6], x[13]]);
  lines($("gripper-chart"), [a.map((x) => x[6]), a.map((x) => x[13])], {
    min: Math.min(0, ...grip),
    max: Math.max(1, ...grip),
  });
  lines($("latency-chart"), [latencies.map((x) => x.ms)], {
    labels: ["较早", "最近"],
  });
}
function renderPrediction() {
  const step = Number($("step").value);
  $("step-label").textContent =
    `第 ${String(step + 1).padStart(2, "0")} / 50 步 · t = ${(step / 30).toFixed(3)} s`;
  $("trajectory-empty").hidden = !!prediction;
  $("view-snapshot").disabled = !prediction;
  if (prediction) {
    const a = prediction.actions[step];
    $("prediction-details").textContent = `L  xyz ${a
      .slice(0, 3)
      .map((v) => fmt(v))
      .join(" / ")}   夹爪 ${fmt(a[6])}\nR  xyz ${a
      .slice(7, 10)
      .map((v) => fmt(v))
      .join(
        " / ",
      )}   夹爪 ${fmt(a[13])}\n相对观察最大位移 L/R: ${prediction.max_displacement_m.map((v) => `${(v * 1000).toFixed(1)} mm`).join(" / ")}`;
    $("prediction-id").textContent =
      `${historyRun || historyPrediction ? "历史" : "最近"}预测 ${prediction.id.slice(0, 10)} · ${prediction.discarded ? "已丢弃" : new Date(prediction.created_unix * 1000).toLocaleTimeString()}`;
    $("prediction-validity").textContent = "数值有效 · 可执行性未验证";
    $("gripper-status").textContent = prediction.gripper_in_range
      ? "50步夹爪目标均在 [0, 1] 内"
      : "存在超出 [0, 1] 的夹爪目标；未裁剪";
    $("gripper-status").classList.toggle("bad", !prediction.gripper_in_range);
    $("rpc-value").textContent = `${(prediction.rpc_s * 1000).toFixed(0)} ms`;
    $("latency-note").textContent =
      `本次模型 ${Math.round(prediction.model_s * 1000)} ms · 观察至结果 ${Math.round(prediction.observation_age_s * 1000)} ms${prediction.startup_sample ? " · 首次请求" : ""}`;
  }
  draw();
}
async function loadPrediction(id, force = false) {
  if (!id || (!force && id === predictionID) || id === loadingPredictionID)
    return;
  loadingPredictionID = id;
  const version = ++fetchVersion;
  try {
    const p = await api(`/predictions/${id}`);
    if (version !== fetchVersion) return;
    prediction = p;
    predictionID = id;
    if (!seenPredictions.has(id)) {
      seenPredictions.add(id);
      latencies.push({ ms: p.rpc_s * 1000 });
      if (latencies.length > 100) latencies.shift();
      if (seenPredictions.size > 1000) seenPredictions.clear();
    }
    renderPrediction();
    if (view === "snapshot") updateImages();
  } catch (e) {
    message(e.message);
  } finally {
    if (loadingPredictionID === id) loadingPredictionID = null;
  }
}
function renderEvents(events) {
  $("event-log").textContent =
    events
      .slice(-60)
      .reverse()
      .map(
        (e) =>
          `${new Date(e.time_unix * 1000).toLocaleTimeString()}  ${e.type}  ${JSON.stringify(e.payload)}`,
      )
      .join("\n") || "尚无事件";
}
function apply(data) {
  if (data.run?.id !== status?.run?.id && !historyRun && !historyPrediction) {
    ++fetchVersion;
    loadingPredictionID = null;
    prediction = null;
    predictionID = null;
    latencies.length = 0;
    $("rpc-value").textContent = "— ms";
    $("prediction-id").textContent = "尚无预测";
    $("prediction-details").textContent =
      "绝对 TCP / 夹爪目标 · 预测步长 1/30 s · 非实际执行进度";
    $("gripper-status").textContent = "等待预测";
    $("latency-note").textContent = "首次请求与稳态分开标记";
    renderPrediction();
    if (view === "snapshot") setView("live");
  }
  status = data;
  if (data.model_manager) renderModelManager(data.model_manager);
  if (
    gripperRestartPending?.run?.id &&
    data.gripper_restart?.run?.id === gripperRestartPending.run.id
  )
    gripperRestartPending = null;
  const restart = gripperRestartState();
  $("restart-gripper").textContent = [
    "UNCERTAIN",
    "INTERRUPTED_REVIEW",
  ].includes(restart?.run?.state)
    ? "核对夹爪状态"
    : restart?.blocked
      ? "初始目标下发中…"
      : "夹爪回初始位";
  $("restart-gripper-status").textContent =
    (restart?.run?.operation === "initial_target_v1"
      ? restart.run.message
      : null) ||
    (restart?.enabled
      ? "显式下发左右初始目标0；不再执行电机失能／使能。"
      : "模拟页面不操作真机夹爪。");
  if (
    restart?.run?.operation === "initial_target_v1" &&
    restart.run.after?.actual?.length === 2
  ) {
    const actual = restart.run.after.actual;
    $("restart-gripper-status").textContent +=
      ` · 确认时实际反馈：左${Number(actual[0]).toFixed(3)}，右${Number(actual[1]).toFixed(3)}`;
  }
  if (homePending?.run?.id && data.home?.run?.id === homePending.run.id)
    homePending = null;
  const home = homeState();
  $("home").textContent = ["UNCERTAIN", "INTERRUPTED_REVIEW"].includes(
    home?.run?.state,
  )
    ? "核对HOME状态"
    : home?.blocked
      ? "HOME复位中…"
      : "HOME · 双臂复位";
  $("home-status").textContent =
    home?.run?.message ||
    (home?.enabled
      ? "双臂复位；完成后可继续推理。运行中的闭环需先停止。"
      : "模拟页面不执行真机HOME。");
  renderClosedLoop(data.closed_loop, data);
  $("source-tag").textContent =
    data.source === "demo" ? "模拟数据 / DEMO" : "真实 ROS 输入";
  $("source-tag").classList.toggle("bad", data.source === "demo");
  $("model-status").textContent = data.model.healthy ? "在线" : "未就绪";
  $("model-status").className = data.model.healthy ? "good" : "bad";
  $("input-status").textContent = data.input.ready ? "输入有效" : "等待 / 异常";
  $("input-status").className = data.input.ready ? "good" : "bad";
  $("input-age").textContent =
    `接收偏差 ${fmt(data.input.skew_s, 3)} s · 曝光同步未验证`;
  $("run-status").textContent = data.run?.state || "IDLE";
  $("run-count").textContent = data.run
    ? `${data.run.count} 次结果 · ${data.run.discarded} 次丢弃`
    : "尚未运行";
  $("p95-value").textContent =
    data.metrics.p95_s == null
      ? "— ms"
      : `${Math.round(data.metrics.p95_s * 1000)} ms`;
  $("sample-count").textContent =
    `最近 ${data.metrics.steady_samples} 个稳态样本`;
  $("result-age").textContent =
    historyRun || historyPrediction
      ? "历史结果 · 非当前推理"
      : data.latest_age_s == null
        ? "尚无结果"
        : `结果距今 ${data.latest_age_s.toFixed(1)} s · 仅预测`;
  const reasons = [
    ...(data.storage_error ? [`日志错误: ${data.storage_error}`] : []),
    ...(!data.model.healthy ? [`模型: ${data.model.error}`] : []),
    ...data.input.reasons,
    ...(data.input.image_warning ? [data.input.image_warning] : []),
    ...(data.input.transport?.warning ? [data.input.transport.warning] : []),
  ];
  $("banner").textContent = reasons.length
    ? reasons.join("；")
    : `${data.source === "demo" ? "模拟模式：所有画面与预测均为合成数据。" : "输入校验通过。"} ${data.run?.reason || "仅单次推理展示预测；开始连续将执行真机闭环。"}`;
  $("banner").className = `notice ${reasons.length ? "error" : "ok"}`;
  feedback(data.state);
  if (!historyRun) renderEvents(data.events);
  if (!historyRun && !historyPrediction && data.latest_prediction_id)
    loadPrediction(
      data.latest_prediction_id,
      Boolean(
        prediction?.id === data.latest_prediction_id &&
          data.latest_prediction_revision &&
          prediction?.offline_ik?.completed_unix !==
            data.latest_prediction_revision,
      ),
    );
  lockButtons();
  for (const [key] of cameras) {
    const el = $(`camera-${key}`);
    el.classList.toggle("stale", view === "live" && !data.input.ready);
    el.querySelector(".camera-state").textContent =
      view === "snapshot"
        ? "SNAPSHOT"
        : data.input.ready
          ? "LIVE"
          : "STALE / WAITING";
  }
  if (!prediction) draw();
  $("view-checked").disabled = !data.ik?.latest_checked_prediction_id;
}
function setConnected(value) {
  connected = value;
  $("connection").textContent = value ? "● 已连接" : "● 连接中断，正在重连";
  $("connection").classList.toggle("bad", !value);
  lockButtons();
}
function pageVisibility() {
  return {
    visibility_state: document.visibilityState,
    has_focus: document.hasFocus(),
  };
}
function reportPageState(event) {
  if (socket?.readyState === WebSocket.OPEN && socket.connectionId) {
    socket.send(
      JSON.stringify({
        kind: "page_state",
        connection_id: socket.connectionId,
        event,
        ...pageVisibility(),
      }),
    );
  }
}
function connect() {
  const ws = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/v1/events`,
  );
  socket = ws;
  let lastSequence = 0;
  let lastAckAt = null;
  ws.onopen = () => {
    if (socket === ws) setConnected(true);
  };
  ws.onmessage = (e) => {
    if (socket !== ws) return;
    try {
      const data = JSON.parse(e.data);
      const challenge = data.heartbeat;
      // Reply before UI rendering. No setInterval or background timer is needed.
      if (
        challenge?.protocol === "server_challenge_v1" &&
        Number.isSafeInteger(challenge.sequence) &&
        challenge.sequence > lastSequence &&
        typeof challenge.connection_id === "string" &&
        typeof challenge.nonce === "string"
      ) {
        ws.connectionId = challenge.connection_id;
        const now = performance.now();
        ws.send(
          JSON.stringify({
            kind: "heartbeat_ack",
            connection_id: challenge.connection_id,
            sequence: challenge.sequence,
            nonce: challenge.nonce,
            since_previous_ack_ms: lastAckAt === null ? null : now - lastAckAt,
            ...pageVisibility(),
          }),
        );
        lastSequence = challenge.sequence;
        lastAckAt = now;
      } else if (
        !challenge &&
        (lastAckAt === null || performance.now() - lastAckAt >= 1000)
      ) {
        // Rolling upgrade compatibility with an older server, also event-driven.
        ws.send("ping");
        lastAckAt = performance.now();
      }
      apply(data);
    } catch (error) {
      message(error.message);
    }
  };
  ws.onclose = (event) => {
    if (ws.connectionId) {
      api("/connection-events", {
        method: "POST",
        body: JSON.stringify({
          connection_id: ws.connectionId,
          code: event.code,
          reason: event.reason.slice(0, 160),
          was_clean: event.wasClean,
          ...pageVisibility(),
        }),
      }).catch(() => {});
    }
    if (socket !== ws) return;
    setConnected(false);
    setTimeout(async () => {
      if (socket !== ws) return;
      try {
        const session = await api("/session");
        csrf = session.csrf_token;
        initControlValidation(api);
        initClosedLoop(api);
      } catch (error) {
        message(`重新建立会话：${error.message}`);
      }
      connect();
    }, 2000);
  };
  ws.onerror = () => ws.close();
}
document.addEventListener("visibilitychange", () =>
  reportPageState("visibilitychange"),
);
for (const event of ["focus", "blur", "pagehide", "pageshow"])
  window.addEventListener(event, () => reportPageState(event));
function updateImages() {
  for (const [key] of cameras) {
    const img = $(`camera-${key}`).querySelector("img");
    if (view === "snapshot" && prediction) {
      const url = `/api/v1/predictions/${prediction.id}/images/${key}`;
      if (img.dataset.url !== url) {
        img.src = url;
        img.dataset.url = url;
      }
    } else if (view === "live" && connected && status?.snapshot_id) {
      img.src = `/api/v1/images/${key}?v=${encodeURIComponent(status.snapshot_id)}`;
      img.dataset.url = "";
    }
  }
  $("camera-mode").textContent =
    view === "snapshot" && prediction
      ? `输入快照 ${prediction.snapshot_id} · 固定帧`
      : "实时输入 · 5 Hz 预览";
}
setInterval(() => {
  if (!document.hidden && view === "live") updateImages();
}, 200);
async function start(kind) {
  busy = true;
  message("");
  lockButtons();
  try {
    const value = $("seed").value;
    const seed = value === "" ? null : Number(value);
    if (seed !== null && (!Number.isSafeInteger(seed) || seed < 0))
      throw Error("随机种子必须是非负安全整数");
    historyRun = "";
    historyPrediction = "";
    $("history-run").value = "";
    $("history-prediction").value = "";
    await api("/runs", {
      method: "POST",
      body: JSON.stringify({
        kind,
        prompt: $("prompt").value,
        seed,
        command_id:
          crypto.randomUUID?.() ||
          `cmd-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      }),
    });
    message("请求已提交");
    await refreshHistory();
  } catch (e) {
    message(e.message);
  } finally {
    busy = false;
    lockButtons();
  }
}
$("restart-gripper").onclick = async () => {
  if ($("restart-gripper").disabled) return;
  gripperRestartSubmitting = true;
  lockButtons();
  try {
    const review = ["UNCERTAIN", "INTERRUPTED_REVIEW"].includes(
      gripperRestartState()?.run?.state,
    );
    const value = await api(
      review ? "/gripper-restart/check" : "/gripper-restart/start",
      {
        method: "POST",
        body: JSON.stringify(
          review
            ? {}
            : {
                command_id: `gripper-initial-${Date.now()}-${Math.random().toString(36).slice(2)}`,
              },
        ),
      },
    );
    gripperRestartPending = value;
    $("restart-gripper-status").textContent =
      value.run?.message || "已核对夹爪状态";
  } catch (error) {
    $("restart-gripper-status").textContent = error.message;
  } finally {
    gripperRestartSubmitting = false;
    lockButtons();
  }
};
$("home").onclick = async () => {
  if ($("home").disabled) return;
  homeSubmitting = true;
  lockButtons();
  try {
    const review = ["UNCERTAIN", "INTERRUPTED_REVIEW"].includes(
      homeState()?.run?.state,
    );
    const value = await api(review ? "/home/check" : "/home/start", {
      method: "POST",
      body: JSON.stringify(
        review
          ? {}
          : {
              command_id: `home-${Date.now()}-${Math.random().toString(36).slice(2)}`,
            },
      ),
    });
    homePending = value;
    $("home-status").textContent = value.run?.message || "已核对HOME状态";
  } catch (error) {
    $("home-status").textContent = error.message;
  } finally {
    homeSubmitting = false;
    lockButtons();
  }
};
$("single").onclick = () => start("single");
$("model-load").onclick = async () => {
  const id = $("model-select").value;
  if (!id) return;
  try {
    renderModelManager({...modelManager, state: "STARTING", error: "正在启动模型服务…"});
    renderModelManager(await api(`/models/${encodeURIComponent(id)}/load`, {method: "POST"}));
  } catch (error) { message(error.message); await refreshModels(); }
};
$("model-stop").onclick = async () => {
  try { renderModelManager(await api("/models/stop", {method: "POST"})); }
  catch (error) { message(error.message); }
};
$("stop").onclick = async () => {
  busy = true;
  lockButtons();
  try {
    await api(`/runs/${status.run.id}/stop`, { method: "POST" });
    message("已停止新请求；在途结果将被丢弃");
  } catch (e) {
    message(e.message);
  } finally {
    busy = false;
    lockButtons();
  }
};
$("prompt").oninput = () => {
  localStorage.setItem("forcevla-prompt", $("prompt").value);
  lockButtons();
};
$("view-live").onclick = () => setView("live");
$("view-snapshot").onclick = () => setView("snapshot");
function setView(value) {
  view = value;
  $("view-live").classList.toggle("selected", value === "live");
  $("view-snapshot").classList.toggle("selected", value === "snapshot");
  updateImages();
}
$("step").oninput = renderPrediction;
$("projection").onchange = draw;
new ResizeObserver(draw).observe($("trajectory"));
async function refreshHistory() {
  try {
    const runs = await api("/runs");
    $("history-run").replaceChildren(
      new Option("当前会话", ""),
      ...runs.map(
        (r) =>
          new Option(
            `${new Date(r.created_unix * 1000).toLocaleString()} · ${r.count}次 · ${r.state}`,
            r.id,
          ),
      ),
    );
    $("history-run").value = historyRun;
  } catch (e) {
    message(e.message);
  }
}
$("refresh-history").onclick = refreshHistory;
$("history-run").onchange = async () => {
  const version = ++historyVersion;
  historyRun = $("history-run").value;
  historyPrediction = "";
  ++fetchVersion;
  $("history-prediction").replaceChildren(new Option("最新预测", ""));
  lockButtons();
  if (!historyRun) {
    predictionID = null;
    if (status?.latest_prediction_id)
      await loadPrediction(status.latest_prediction_id);
    return;
  }
  try {
    const [run, events] = await Promise.all([
      api(`/runs/${historyRun}`),
      api(`/runs/${historyRun}/events`),
    ]);
    if (version !== historyVersion) return;
    renderEvents(events);
    $("history-prediction").replaceChildren(
      ...run.predictions.map(
        (p) =>
          new Option(
            `${new Date(p.created_unix * 1000).toLocaleTimeString()} · ${p.discarded ? "丢弃" : Math.round(p.rpc_s * 1000) + "ms"}`,
            p.id,
          ),
      ),
    );
    if (run.predictions.length) {
      historyPrediction = run.predictions[0].id;
      await loadPrediction(historyPrediction);
    } else {
      prediction = null;
      predictionID = null;
      renderPrediction();
      message("该会话没有已保存预测");
    }
  } catch (e) {
    message(e.message);
  }
};
$("history-prediction").onchange = () => {
  historyPrediction = $("history-prediction").value;
  loadPrediction(historyPrediction);
};
$("export").onclick = () => {
  const id = historyRun || status?.run?.id;
  if (id) location.assign(`/api/v1/runs/${id}/export`);
};
try {
  const session = await api("/session");
  csrf = session.csrf_token;
  initControlValidation(api);
  initClosedLoop(api);
  $("prompt").value =
    localStorage.getItem("forcevla-prompt") || localStorage.getItem("shadow-prompt") || session.default_prompt;
  connect();
  refreshModels();
  refreshHistory();
} catch (e) {
  message(`初始化失败：${e.message}`);
}

$("view-checked").onclick = () => {
  const id = status?.ik?.latest_checked_prediction_id;
  if (id) {
    historyPrediction = id;
    loadPrediction(id, true);
  }
};
new ResizeObserver(() =>
  renderIK(
    prediction,
    Number($("step").value),
    $("projection").value,
    status?.ik,
  ),
).observe($("joint-pose"));

$("follow-latest").onclick = () => {
  historyRun = "";
  historyPrediction = "";
  predictionID = null;
  $("history-run").value = "";
  $("history-prediction").replaceChildren(new Option("最新预测", ""));
  loadPrediction(status?.latest_prediction_id, true);
};
setInterval(() => {
  if (connected && prediction?.offline_ik?.status === "pending")
    loadPrediction(prediction.id, true);
}, 1000);
