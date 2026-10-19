const LEFT = "#64d9cc",
  RIGHT = "#f2aa71";
function prepare(canvas) {
  const r = canvas.getBoundingClientRect(),
    d = devicePixelRatio || 1;
  canvas.width = r.width * d;
  canvas.height = r.height * d;
  const c = canvas.getContext("2d");
  c.scale(d, d);
  c.font = "10px system-ui";
  return [c, r.width, r.height];
}
export function trajectory(canvas, prediction, state, step, view) {
  const [c, w, h] = prepare(canvas);
  const pts = prediction ? prediction.actions : [];
  const observation = prediction?.state || state || [];
  const series = [
    pts.map((p) => p.slice(0, 3)),
    pts.map((p) => p.slice(7, 10)),
  ];
  const actual = [observation.slice(0, 3), observation.slice(13, 16)].filter(
    (p) => p.length === 3 && p.every(Number.isFinite),
  );
  const all = [...series.flat(), ...actual];
  if (!all.length) return;
  const project = ([x, y, z]) =>
    view === "xy"
      ? [x, y]
      : view === "xz"
        ? [x, z]
        : view === "yz"
          ? [y, z]
          : [(x - y) * 0.707, z * 0.82 - (x + y) * 0.35];
  const projected = all.map(project),
    lo = [0, 1].map((i) => Math.min(...projected.map((p) => p[i]))),
    hi = [0, 1].map((i) => Math.max(...projected.map((p) => p[i])));
  const range = [0, 1].map((i) => Math.max(hi[i] - lo[i], 0.15));
  const scale = Math.min((w - 110) / range[0], (h - 100) / range[1]) * 0.8;
  const mid = [0, 1].map((i) => (lo[i] + hi[i]) / 2);
  const px = (p) => {
    const q = project(p);
    return [w / 2 + (q[0] - mid[0]) * scale, h / 2 - (q[1] - mid[1]) * scale];
  };
  c.strokeStyle = "#23334a";
  c.lineWidth = 1;
  for (let x = 35; x < w; x += 40) {
    c.beginPath();
    c.moveTo(x, 25);
    c.lineTo(x, h - 35);
    c.stroke();
  }
  for (let y = 25; y < h - 25; y += 40) {
    c.beginPath();
    c.moveTo(30, y);
    c.lineTo(w - 25, y);
    c.stroke();
  }
  c.fillStyle = "#657c9c";
  c.fillText(
    view === "iso" ? "斜投影 · X / Y / Z" : `${view.toUpperCase()} 坐标投影`,
    18,
    23,
  );
  c.fillText(`网格屏幕尺度 ≈ ${(40 / scale).toFixed(3)} m`, 18, 39);
  series.forEach((s, i) => {
    if (!s.length) return;
    c.strokeStyle = i ? RIGHT : LEFT;
    c.lineWidth = 2;
    c.beginPath();
    s.forEach((p, j) => {
      const [x, y] = px(p);
      j ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    c.stroke();
    s.forEach((p, j) => {
      if (j % 5) return;
      const [x, y] = px(p);
      c.fillStyle = i ? RIGHT : LEFT;
      c.beginPath();
      c.arc(x, y, 2, 0, Math.PI * 2);
      c.fill();
    });
    const [x, y] = px(s[step]);
    c.fillStyle = i ? RIGHT : LEFT;
    c.beginPath();
    c.arc(x, y, 5, 0, Math.PI * 2);
    c.fill();
    c.fillText(`${i ? "R" : "L"} ${step + 1}`, x + 9, y - 7);
  });
  actual.forEach((p, i) => {
    const [x, y] = px(p);
    c.strokeStyle = i ? RIGHT : LEFT;
    c.lineWidth = 2;
    c.beginPath();
    c.arc(x, y, 7, 0, Math.PI * 2);
    c.stroke();
    c.fillStyle = i ? RIGHT : LEFT;
    c.fillText(`${i ? "R" : "L"} 观察`, x + 10, y + 15);
  });
}
export function lines(
  canvas,
  series,
  { labels = [], min = 0, max = null, colors = [LEFT, RIGHT] } = {},
) {
  const [c, w, h] = prepare(canvas);
  const values = series.flat();
  max = max ?? Math.max(1, ...values) * 1.15;
  const x0 = 46,
    y0 = 17,
    cw = w - 65,
    ch = h - 42;
  for (let k = 0; k <= 3; k++) {
    let y = y0 + (ch * k) / 3;
    c.strokeStyle = "#253145";
    c.beginPath();
    c.moveTo(x0, y);
    c.lineTo(w - 15, y);
    c.stroke();
    c.fillStyle = "#6e829f";
    c.fillText(
      (max - ((max - min) * k) / 3).toFixed(max > 10 ? 0 : 2),
      8,
      y + 3,
    );
  }
  series.forEach((s, i) => {
    c.strokeStyle = colors[i % colors.length];
    c.lineWidth = 1.7;
    c.beginPath();
    s.forEach((v, j) => {
      const x = x0 + (j * cw) / Math.max(1, s.length - 1),
        y = y0 + ch - ((v - min) / (max - min)) * ch;
      j ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    c.stroke();
    if (s.length === 1) {
      c.fillStyle = colors[i % colors.length];
      c.beginPath();
      c.arc(x0, y0 + ch - ((s[0] - min) / (max - min)) * ch, 3, 0, Math.PI * 2);
      c.fill();
    }
  });
  c.fillStyle = "#6e829f";
  c.fillText(labels[0] || "0", x0, h - 8);
  c.fillText(labels[1] || "49", w - 65, h - 8);
}

export function joints(canvas, seed, predicted, view = "iso", failed = false) {
  const [c, w, h] = prepare(canvas);
  const all = [
    ...(seed?.left || []),
    ...(seed?.right || []),
    ...(predicted?.left || []),
    ...(predicted?.right || []),
    [0, 0, 0],
  ];
  if (all.length < 2) {
    c.fillStyle = "#8796ad";
    c.fillText("等待关节快照与离线检查", 20, 35);
    return;
  }
  const project = ([x, y, z]) =>
    view === "xy"
      ? [x, y]
      : view === "xz"
        ? [x, z]
        : view === "yz"
          ? [y, z]
          : [(x - y) * 0.707, z * 0.82 - (x + y) * 0.35];
  const points = all.map(project),
    lo = [0, 1].map((i) => Math.min(...points.map((p) => p[i]))),
    hi = [0, 1].map((i) => Math.max(...points.map((p) => p[i])));
  const scale = Math.min(
    (w - 100) / Math.max(0.2, hi[0] - lo[0]),
    (h - 70) / Math.max(0.2, hi[1] - lo[1]),
  );
  const px = (p) => {
    const q = project(p);
    return [
      w / 2 + (q[0] - (hi[0] + lo[0]) / 2) * scale,
      h / 2 - (q[1] - (hi[1] + lo[1]) / 2) * scale,
    ];
  };
  const stroke = (chain, color, width, dashed = false) => {
    if (!chain?.length) return;
    c.strokeStyle = color;
    c.fillStyle = color;
    c.lineWidth = width;
    c.setLineDash(dashed ? [4, 4] : []);
    c.beginPath();
    chain.forEach((p, i) => {
      const [x, y] = px(p);
      i ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    c.stroke();
    c.setLineDash([]);
    chain.forEach((p, i) => {
      const [x, y] = px(p);
      c.beginPath();
      c.arc(x, y, i === chain.length - 1 ? 5 : 3, 0, 2 * Math.PI);
      c.fill();
    });
  };
  stroke(
    [
      [0, 0, 0],
      [0, 0, 1.121],
    ],
    "#45536a",
    6,
  );
  for (const side of ["left", "right"])
    stroke(seed?.[side], "#6c7d94", 2, true);
  stroke(predicted?.left, failed ? "#ff878c" : LEFT, 3);
  stroke(predicted?.right, failed ? "#e7a267" : RIGHT, 3);
  c.fillStyle = "#8796ad";
  c.fillText("虚线：观察时姿态  ·  实线：该步求解姿态", 18, h - 13);
}
