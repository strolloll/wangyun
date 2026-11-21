const $ = (id) => document.getElementById(id);

const state = {
  bundle: null,
  params: null,
  index: 0,
  playing: false,
  timer: null,
  hover: null,
};

function readParams() {
  const p = {};
  for (const key of ["n", "modules", "ticks", "seed", "probe", "need", "frac"]) {
    p[key] = $(key).value;
  }
  return p;
}

function qs(params) {
  return new URLSearchParams(params).toString();
}

function thumbSize() {
  return Number($("thumbsize")?.value) || 320;
}

async function run() {
  const params = readParams();
  $("run").disabled = true;
  $("status").textContent = "在跑…";
  const t0 = performance.now();
  try {
    const res = await fetch("/api/bundle?" + qs(params));
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    state.bundle = data;
    state.params = params;
    state.index = 0;
    state.hover = null;

    const dl = $("bundle");
    dl.href = "/api/bundle?" + qs(params);
    dl.download = `wangyun-n${params.n}-seed${params.seed}.json`;

    const scrub = $("scrub");
    scrub.max = Math.max(0, data.frames.length - 1);
    scrub.value = 0;
    $("status").textContent =
      `${data.frames.length} 帧 / ${data.events.length} 次变化 · ` +
      `${Math.round(performance.now() - t0)} ms`;

    renderEvents();
    draw();
  } catch (err) {
    $("status").textContent = "出错：" + err.message;
    console.error(err);
  } finally {
    $("run").disabled = false;
  }
}

function nearestFrame(tick) {
  const frames = state.bundle.frames;
  let best = 0;
  for (let i = 0; i < frames.length; i++) {
    if (Math.abs(frames[i].t - tick) < Math.abs(frames[best].t - tick)) best = i;
  }
  return best;
}

function renderEvents() {
  const ol = $("events");
  ol.textContent = "";
  for (const ev of state.bundle.events) {
    const li = document.createElement("li");
    li.className = ev.kind;
    const row = document.createElement("div");
    row.className = "ev-row";
    const idx = nearestFrame(ev.tick);
    const shot = document.createElement("a");
    shot.className = "shot";
    shot.dataset.index = String(idx);
    shot.href = `/api/png?${qs(state.params)}&index=${idx}&size=${thumbSize()}`;
    shot.target = "_blank";
    shot.rel = "noreferrer";
    shot.title = "原图（Pillow）";
    shot.addEventListener("click", (e) => e.stopPropagation());
    const img = document.createElement("img");
    img.className = "thumb";
    img.loading = "lazy";
    img.alt = `t=${ev.tick}`;
    img.src = shot.href;
    shot.appendChild(img);
    const text = document.createElement("span");
    text.className = "ev-text";
    text.textContent = `t=${ev.tick}  ${ev.text}`;
    row.append(shot, text);
    li.appendChild(row);
    li.title = `k ${ev.k_before} → ${ev.k_after}`;
    li.addEventListener("click", () => {
      stop();
      state.index = nearestFrame(ev.tick);
      $("scrub").value = state.index;
      draw();
    });
    ol.appendChild(li);
  }
  if (!state.bundle.events.length) {
    const li = document.createElement("li");
    li.textContent = "这一跑没出现划分变化（视角太粗或太短）";
    ol.appendChild(li);
  }
}

function refreshThumbs() {
  if (!state.bundle || !state.params) return;
  const size = thumbSize();
  for (const shot of document.querySelectorAll("#events .shot")) {
    shot.href = `/api/png?${qs(state.params)}&index=${shot.dataset.index}&size=${size}`;
    const img = shot.firstElementChild;
    if (img) img.src = shot.href;
  }
}

function draw() {
  const canvas = $("view");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#061010";
  ctx.fillRect(0, 0, W, H);

  if (!state.bundle || !state.bundle.frames.length) {
    ctx.fillStyle = "#7f9a96";
    ctx.font = "14px monospace";
    ctx.fillText("点「跑一遍」开始", 24, 32);
    return;
  }

  const f = state.bundle.frames[state.index];
  const k = f.k;
  const cx = W / 2;
  const cy = H / 2 + 6;
  const R = Math.min(W, H) * 0.42;
  const xy = (i) => [cx + f.pos[i][0] * R, cy - f.pos[i][1] * R];

  // 群间强度
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const w = f.strength[i][j];
      if (w <= 0.02) continue;
      const [x1, y1] = xy(i);
      const [x2, y2] = xy(j);
      ctx.strokeStyle = `rgba(127, 214, 208, ${Math.min(0.5, 0.18 + 0.8 * w)})`;
      ctx.lineWidth = 0.6 + 1.6 * w;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }

  // 群：色相来自 bundle（按成员记路），半径只表示人数
  for (let i = 0; i < k; i++) {
    const g = f.groups[i];
    const [x, y] = xy(i);
    const r = Math.max(6, Math.min(44, 5 + 5 * Math.sqrt(g.count)));
    const hue = Math.round(g.hue * 360);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    if (g.settled) {
      ctx.fillStyle = `hsl(${hue} 58% 64%)`;
      ctx.globalAlpha = 0.9;
      ctx.fill();
      ctx.globalAlpha = 1;
    } else {
      ctx.fillStyle = `hsl(${hue} 58% 64%)`;
      ctx.globalAlpha = 0.28;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = `hsl(${hue} 58% 74%)`;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // 角上与底部读数
  ctx.fillStyle = "#d8e8e4";
  ctx.font = "13px monospace";
  ctx.fillText(`t=${f.t}  k=${k}`, 18, 26);
  ctx.fillStyle = f.hue_moved > 0 ? "#f0a048" : "#7f9a96";
  ctx.fillText(`色相移动 ${Number(f.hue_moved).toFixed(3)}`, W - 190, 26);
  ctx.fillStyle = "#7f9a96";
  ctx.font = "12px monospace";
  const m = state.bundle.meta;
  ctx.fillText(
    `n=${m.n}  seed=${m.seed}  视角=${m.frac}  采样=${m.probe} tick`,
    18,
    H - 16
  );

  $("frameinfo").textContent = `帧 ${state.index + 1}/${state.bundle.frames.length}  t=${f.t}`;
  const svg = $("svg");
  svg.href = `/api/svg?${qs(state.params)}&index=${state.index}`;
  const size = Number($("size")?.value) || 760;
  $("png").href = `/api/png?${qs(state.params)}&index=${state.index}&size=${size}`;
  drawBand();
}

function drawBand() {
  const canvas = $("band");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!state.bundle) return;

  const meta = state.bundle.meta || {};
  const readings = state.bundle.readings || {};
  const ts = Array.isArray(readings.t) ? readings.t : [];
  const ks = Array.isArray(readings.k) ? readings.k : [];
  const frames = state.bundle.frames || [];
  const events = state.bundle.events || [];
  const total = Math.max(1, Number(meta.ticks) || 0);
  const left = 10;
  const right = 10;
  const xOf = (t) => left + (t / total) * (W - left - right);

  const maxK = Math.max(1, ...ks);
  const kTop = 10;
  const kBottom = 52;
  const hueBase = H - 8;
  const hueTop = H - 46;
  const moves = frames.map((f) => Number(f.hue_moved) || 0);
  const maxMove = Math.max(1e-9, ...moves);
  const xNow = frames[state.index] ? xOf(frames[state.index].t) : left;

  for (const ev of events) {
    const x = xOf(ev.tick);
    ctx.strokeStyle = ev.kind === "merge" ? "rgba(240,160,72,0.22)" : "rgba(96,208,214,0.22)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, kTop);
    ctx.lineTo(x, hueBase);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(216,232,228,0.12)";
  ctx.beginPath();
  ctx.moveTo(left, hueBase);
  ctx.lineTo(W - right, hueBase);
  ctx.stroke();

  if (ts.length && ks.length) {
    ctx.strokeStyle = "rgba(127,214,208,0.85)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ts.forEach((t, i) => {
      const x = xOf(t);
      const y = kBottom - (ks[i] / maxK) * (kBottom - kTop);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  for (let i = 0; i < frames.length; i++) {
    const m = moves[i];
    if (!m) continue;
    const x = xOf(frames[i].t);
    const h = (m / maxMove) * (hueBase - hueTop);
    ctx.strokeStyle = "rgba(201,167,224,0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, hueBase);
    ctx.lineTo(x, hueBase - h);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(240,160,72,0.9)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(xNow, kTop - 4);
  ctx.lineTo(xNow, hueBase + 4);
  ctx.stroke();

  ctx.fillStyle = "rgba(216,232,228,0.55)";
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText(`k(t) 最高 ${maxK}`, left, kTop + 16);
  ctx.fillText("色相移动", left, hueBase - 5);

  if (state.hover) {
    const h = state.hover;
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "rgba(216,232,228,0.55)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(h.x, kTop - 4);
    ctx.lineTo(h.x, hueBase + 4);
    ctx.stroke();
    ctx.restore();

    const kind = h.ev ? (h.ev.kind === "merge" ? "并" : "分") : "";
    const label = h.ev
      ? `t=${h.t}  k=${h.k}  色相移动=${h.moved.toFixed(3)}  · ${kind} ${h.ev.k_before}→${h.ev.k_after}`
      : `t=${h.t}  k=${h.k}  色相移动=${h.moved.toFixed(3)}`;
    ctx.font = "11px system-ui, sans-serif";
    const tw = ctx.measureText(label).width;
    const tx = Math.min(Math.max(h.x + 8, left), W - tw - 10);
    ctx.fillStyle = "rgba(4,16,16,0.92)";
    ctx.fillRect(tx - 6, 2, tw + 12, 18);
    ctx.fillStyle = "rgba(240,220,180,0.95)";
    ctx.fillText(label, tx, 15);
  }
}

function play() {
  if (state.playing) return stop();
  if (!state.bundle) return;
  state.playing = true;
  $("play").textContent = "停";
  state.timer = setInterval(() => {
    if (!state.bundle) return stop();
    state.index += 1;
    if (state.index >= state.bundle.frames.length) return stop();
    $("scrub").value = state.index;
    draw();
  }, 150);
}

function stop() {
  state.playing = false;
  $("play").textContent = "播放";
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}

async function loadPlugins() {
  try {
    const res = await fetch("/api/plugins");
    const data = await res.json();
    const ul = $("plugins");
    ul.textContent = "";
    for (const [kind, rows] of Object.entries(data)) {
      for (const [name, doc] of Object.entries(rows)) {
        const li = document.createElement("li");
        li.textContent = `${kind}/${name}`;
        li.title = doc;
        ul.appendChild(li);
      }
    }
  } catch (err) {
    console.error(err);
  }
}

$("run").addEventListener("click", run);
$("play").addEventListener("click", play);
$("size").addEventListener("input", (e) => {
  $("sizeval").textContent = e.target.value;
  if (state.bundle) draw();
});
let thumbTimer = null;
$("thumbsize").addEventListener("input", (e) => {
  $("thumbsizeval").textContent = e.target.value;
  clearTimeout(thumbTimer);
  thumbTimer = setTimeout(refreshThumbs, 250);
});
$("band").addEventListener("mousemove", (e) => {
  if (!state.bundle) return;
  const canvas = $("band");
  const rect = canvas.getBoundingClientRect();
  const W = canvas.width;
  const left = 10;
  const right = 10;
  const x = (e.clientX - rect.left) * (W / rect.width);
  const total = Math.max(1, Number(state.bundle.meta?.ticks) || 0);
  const t = Math.max(0, Math.min(total, ((x - left) / (W - left - right)) * total));

  const readings = state.bundle.readings || {};
  const ts = Array.isArray(readings.t) ? readings.t : [];
  const ks = Array.isArray(readings.k) ? readings.k : [];
  let ri = 0;
  for (let i = 0; i < ts.length; i++) {
    if (Math.abs(ts[i] - t) < Math.abs(ts[ri] - t)) ri = i;
  }
  const frames = state.bundle.frames || [];
  let fi = 0;
  for (let i = 0; i < frames.length; i++) {
    if (Math.abs(frames[i].t - t) < Math.abs(frames[fi].t - t)) fi = i;
  }
  const moved = frames.length ? Number(frames[fi].hue_moved) || 0 : 0;
  const tol = (total / (W - left - right)) * 8;
  let ev = null;
  for (const item of state.bundle.events || []) {
    if (Math.abs(item.tick - t) <= tol) {
      ev = item;
      break;
    }
  }
  state.hover = { x, t: Math.round(t), k: ts.length ? ks[ri] : "?", moved, ev };
  drawBand();
});
$("band").addEventListener("mouseleave", () => {
  state.hover = null;
  drawBand();
});
$("scrub").addEventListener("input", (e) => {
  stop();
  state.index = Number(e.target.value);
  draw();
});
document.addEventListener("keydown", (e) => {
  if (!state.bundle) return;
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    stop();
    state.index = Math.max(0, Math.min(state.bundle.frames.length - 1,
      state.index + (e.key === "ArrowLeft" ? -1 : 1)));
    $("scrub").value = state.index;
    draw();
  }
});

$("quit").addEventListener("click", async () => {
  if (!confirm("退出望云？本地服务会停掉，这页可以关掉。")) return;
  stop();
  try {
    await fetch("/api/quit", { method: "POST" });
    $("byebye").textContent = "已退出，关掉这页就行。";
  } catch (err) {
    $("byebye").textContent = "没送到：服务可能已经停了。";
  }
});

loadPlugins();
run();
