/**
 * wangyun-tools · 把望云的核心读数接进 DSH
 *
 * Host 半边只做两件事：给模型两个工具 `wy_observe`（总览）和 `wy_frame`（某一帧）。
 * 它不自己算——望云的核心（`python -m wy.core`）是纯 numpy 的，插件只负责把参数递过去、
 * 把回来的 bundle 压成一段能读的字。图、网页、别的软件都吃同一份 bundle。
 *
 * 没有守护进程，没有轮询：不调用就不动。
 * @module wangyun-tools
 */
import { execFile } from 'node:child_process';
import path from 'node:path';

const DEFAULTS = {
  /* 本机 runtime 里的 python；换机器/换解释器在 cordis.patch.yml 里改。 */
  python: 'C:\\Users\\CS TAO\\.dsh\\dsh-runtimes\\dsh-primary-runtime\\dependencies\\python\\python.exe',
  /* 望云仓库根：`python -m wy.core` 要能在这儿 import 到 wy。 */
  workspace: 'D:\\望云',
  timeoutMs: 120_000,
};

function normalize(raw) {
  const cfg = { ...DEFAULTS, ...(raw && typeof raw === 'object' ? raw : {}) };
  cfg.python = String(cfg.python || DEFAULTS.python);
  cfg.workspace = path.resolve(String(cfg.workspace || DEFAULTS.workspace));
  cfg.timeoutMs = Math.min(Math.max(Number(cfg.timeoutMs) || DEFAULTS.timeoutMs, 5_000), 600_000);
  return cfg;
}

/** 跑一次核心命令行，拿回 stdout 里的 bundle JSON。 */
function runCore(config, args) {
  const argv = [
    '-m', 'wy.core',
    '--n', String(args.n),
    '--modules', String(args.modules),
    '--ticks', String(args.ticks),
    '--seed', String(args.seed),
    '--frac', String(args.frac),
    '--probe', String(args.probe),
    '--need', String(args.need),
  ];
  if (args.members) argv.push('--members');
  if (args.out) argv.push('--out', String(args.out));

  return new Promise((resolve, reject) => {
    execFile(
      config.python,
      argv,
      {
        cwd: config.workspace,
        timeout: config.timeoutMs,
        maxBuffer: 64 * 1024 * 1024,
        windowsHide: true,
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      },
      (err, stdout, stderr) => {
        if (err) {
          const tail = String(stderr || err.message || '').trim().split(/\r?\n/).slice(-6).join('\n');
          reject(Object.assign(new Error(tail || '望云核心没有正常结束'), { code: err.code || 'core_failed' }));
          return;
        }
        try {
          resolve(JSON.parse(stdout));
        } catch {
          reject(Object.assign(new Error(`核心返回的不是 JSON：${String(stdout).slice(0, 200)}`), { code: 'bad_json' }));
        }
      },
    );
  });
}

/** 把 bundle 压成有界的一段字：读数、事件、末帧。 */
function summarize(bundle, note) {
  const meta = bundle.meta || {};
  const readings = bundle.readings || {};
  const xs = Array.isArray(readings.t) ? readings.t : [];
  const ks = Array.isArray(readings.k) ? readings.k : [];
  const lines = [];

  lines.push(
    `望云读数：n=${meta.n} · ${meta.modules} 模块 · ${meta.ticks} tick · seed=${meta.seed} · 视角阈值 ${meta.frac} · 稳定位数门槛 ${meta.need}`,
  );

  if (ks.length) {
    const heads = ks.slice(0, 4).join(' ');
    const last = ks[ks.length - 1];
    const most = Math.max(...ks);
    lines.push(`类型数 k(t)@2%：${heads} … → ${last}（最高 ${most}，${xs.length} 个点）`);
  }

  const frames = Array.isArray(bundle.frames) ? bundle.frames : [];
  const frameKs = frames.map((f) => f.k);
  if (frameKs.length) {
    const fh = frameKs.slice(0, 8).join(' ');
    lines.push(`划分（每 ${meta.probe || '?'} tick 一个采样帧）：${fh}${frameKs.length > 8 ? ' …' : ''}（k=${frameKs[frameKs.length - 1]}）`);
  }

  const defect = Array.isArray(readings.defect) ? readings.defect : [];
  const stable = Array.isArray(readings.stable) ? readings.stable : [];
  if (defect.length >= 2) {
    lines.push(`违约读数（对数）：${defect[0]} → ${defect[defect.length - 1]}`);
  }
  if (stable.length >= 2) {
    lines.push(`稳定位数（均值）：${stable[0]} → ${stable[stable.length - 1]}`);
  }

  const events = Array.isArray(bundle.events) ? bundle.events : [];
  if (events.length) {
    const shown = events.slice(0, 8).map((e) => `t=${e.tick} ${e.kind} ${e.k_before}→${e.k_after}`);
    lines.push(`划分变化 ${events.length} 次：${shown.join('；')}${events.length > 8 ? ' …' : ''}`);
  } else {
    lines.push('这一跑划分一次都没变。');
  }

  const frame = frames[frames.length - 1];
  if (frame) {
    const groups = (frame.groups || [])
      .slice()
      .sort((a, b) => b.count - a.count)
      .slice(0, 8)
      .map((g) => `${g.count}人/${Math.round((g.hue || 0) * 360)}°${g.settled ? '' : '·未稳'}`);
    lines.push(`末帧 t=${frame.t}（k=${frame.k}）：${groups.join('、')}`);
    if (typeof frame.hue_moved === 'number') {
      lines.push(`末帧色相移动：${frame.hue_moved.toFixed(3)}（换群才换色，§30）`);
    }
  }

  if (note) lines.push(note);
  lines.push('完整 bundle 是 JSON；要落盘就带 out=...，网页端是 `python -m wy serve`。');
  return lines.join('\n');
}

/** 把某一帧压成有界的一段字：群、稳没稳、哪几对群最紧。 */
function frameSummary(bundle, want, note) {
  const meta = bundle.meta || {};
  const frames = Array.isArray(bundle.frames) ? bundle.frames : [];
  if (!frames.length) {
    return `这一跑没有采样帧（ticks=${meta.ticks} 比 probe=${meta.probe} 小）。`;
  }
  let pick = 0;
  for (let i = 0; i < frames.length; i++) {
    if (Math.abs(frames[i].t - want) < Math.abs(frames[pick].t - want)) pick = i;
  }
  const frame = frames[pick];
  const raw = Array.isArray(frame.groups) ? frame.groups : [];
  const groups = raw
    .map((g, i) => ({ g, i }))
    .sort((a, b) => b.g.count - a.g.count);
  const lines = [
    `望云那一帧：t=${frame.t}（你要 t≈${want}）· k=${frame.k} · n=${meta.n} seed=${meta.seed}`,
  ];
  for (const { g, i } of groups.slice(0, 10)) {
    lines.push(
      `  #${g.gid}（第 ${i + 1} 群）：${g.count} 人 · ${Math.round((g.hue || 0) * 360)}°`
        + ` · 稳定位数 ${Number(g.stable || 0).toFixed(1)} · ${g.settled ? '稳' : '未稳'}`,
    );
    if (Array.isArray(g.members) && g.members.length) {
      const shown = g.members.slice(0, 12).join(' ');
      lines.push(`    成员：${shown}${g.members.length > 12 ? ` …（共 ${g.members.length} 人）` : ''}`);
    }
  }
  if (groups.length > 10) lines.push(`  …还有 ${groups.length - 10} 群`);

  const strength = Array.isArray(frame.strength) ? frame.strength : [];
  const pairs = [];
  for (let i = 0; i < strength.length; i++) {
    for (let j = i + 1; j < strength[i].length; j++) {
      const w = Number(strength[i][j]);
      if (w > 0) pairs.push({ i, j, w });
    }
  }
  pairs.sort((a, b) => b.w - a.w);
  if (pairs.length) {
    const fmt = (w) => (w >= 0.1 ? w.toFixed(2) : w.toFixed(3));
    const top = pairs.slice(0, 4).map((p) => `#${raw[p.i]?.gid}–#${raw[p.j]?.gid} ${fmt(p.w)}`);
    lines.push(`最紧的几对群：${top.join('、')}`);
  } else {
    lines.push('这一帧群之间没有正的关系（都松）。');
  }

  if (typeof frame.hue_moved === 'number') {
    lines.push(`这一帧色相移动：${frame.hue_moved.toFixed(3)}（换群才换色，§30）`);
  }
  if (note) lines.push(note);
  return lines.join('\n');
}

function createTool(config) {
  return {
    name: 'wy_observe',
    description:
      '跑一次望云读数，看这个网络长成了什么：分成几群、划分有没有变、末帧谁和谁一伙。'
      + '核心是纯 numpy 的，不画图也能跑。要看动画/诊断带用命令行或网页端；这里只给读数。',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        n: { type: 'integer', minimum: 8, maximum: 2000, description: '人数，默认 128。' },
        modules: { type: 'integer', minimum: 1, maximum: 24, description: '模块数，默认 8。' },
        ticks: { type: 'integer', minimum: 10, maximum: 20000, description: '跑多少 tick，默认 1200。' },
        seed: { type: 'integer', minimum: 0, maximum: 100000, description: '种子，默认 7。' },
        frac: { type: 'number', minimum: 0.0001, maximum: 0.5, description: '划分用的关系比例阈值，默认 0.03。' },
        probe: { type: 'integer', minimum: 2, maximum: 500, description: '每几 tick 记一个采样点，默认 20。' },
        need: { type: 'integer', minimum: 1, maximum: 64, description: '稳定位数门槛，默认 4。' },
        members: { type: 'boolean', description: '每个群带上成员下标（bundle 更大，一般不用）。' },
        out: { type: 'string', description: '把完整 bundle JSON 写到这里（可选，逗号分隔、absolute path）。' },
      },
    },
    timeoutMs: config.timeoutMs,
    output: {
      schema: { type: 'object', additionalProperties: true },
      render: (_args, value) => [{ type: 'text', text: String(value?.text ?? '') }],
    },
    async execute(args = {}) {
      const a = {
        n: Number(args.n) || 128,
        modules: Number(args.modules) || 8,
        ticks: Number(args.ticks) || 1200,
        seed: Number.isFinite(args.seed) ? Number(args.seed) : 7,
        frac: Number(args.frac) || 3e-2,
        probe: Number(args.probe) || 20,
        need: Number(args.need) || 4,
        members: args.members === true,
        out: args.out ? String(args.out) : '',
      };
      try {
        const bundle = await runCore(config, a);
        return { text: summarize(bundle, a.out ? `完整 JSON 落在 ${a.out}` : '') };
      } catch (e) {
        return {
          text: `没跑成（${e?.code || 'error'}）：${String(e?.message || e).slice(-600)}`
            + `\n检查 cordis.patch.yml 里的 python（现在是 ${config.python}）和 workspace（${config.workspace}）。`,
        };
      }
    },
  };
}

/**
 * `wy_frame`：先 `wy_observe` 看哪儿换了群，再用它追问「那一刻是谁和谁」。
 * @param config 归一化后的配置
 */
function createFrameTool(config) {
  return {
    name: 'wy_frame',
    description:
      '看某一 tick 附近的那一帧：分成几群、每群几个人、什么颜色、稳没稳，哪几对群之间最紧。'
      + '先用 wy_observe 看事件，再用它追问那一刻；要具体是谁就带 members=true。',
    parameters: {
      type: 'object',
      additionalProperties: false,
      required: ['tick'],
      properties: {
        tick: { type: 'integer', minimum: 0, maximum: 20000, description: '想看的那一 tick（会给最接近的采样帧）。' },
        n: { type: 'integer', minimum: 8, maximum: 2000, description: '人数，默认 128。' },
        modules: { type: 'integer', minimum: 1, maximum: 24, description: '模块数，默认 8。' },
        ticks: { type: 'integer', minimum: 10, maximum: 20000, description: '跑多少 tick，默认 1200。' },
        seed: { type: 'integer', minimum: 0, maximum: 100000, description: '种子，默认 7。' },
        frac: { type: 'number', minimum: 0.0001, maximum: 0.5, description: '划分用的关系比例阈值，默认 0.03。' },
        probe: { type: 'integer', minimum: 2, maximum: 500, description: '每几 tick 记一个采样点，默认 20。' },
        need: { type: 'integer', minimum: 1, maximum: 64, description: '稳定位数门槛，默认 4。' },
        members: { type: 'boolean', description: '每个群带上成员下标（bundle 更大；只看前几个群的人时用）。' },
        out: { type: 'string', description: '把完整 bundle JSON 写到这里（可选，absolute path）。' },
      },
    },
    timeoutMs: config.timeoutMs,
    output: {
      schema: { type: 'object', additionalProperties: true },
      render: (_args, value) => [{ type: 'text', text: String(value?.text ?? '') }],
    },
    async execute(args = {}) {
      const clampInt = (v, lo, hi, dft) => {
        const x = Number(v);
        return Number.isInteger(x) ? Math.min(Math.max(x, lo), hi) : dft;
      };
      const a = {
        tick: clampInt(args.tick, 0, 20000, 0),
        n: clampInt(args.n, 8, 2000, 128),
        modules: clampInt(args.modules, 1, 24, 8),
        ticks: clampInt(args.ticks, 10, 20000, 1200),
        seed: clampInt(args.seed, 0, 100000, 7),
        frac: (() => {
          const x = Number(args.frac);
          return Number.isFinite(x) ? Math.min(Math.max(x, 1e-4), 0.5) : 3e-2;
        })(),
        probe: clampInt(args.probe, 2, 500, 20),
        need: clampInt(args.need, 1, 64, 4),
        members: args.members === true,
        out: args.out ? String(args.out) : '',
      };
      try {
        const bundle = await runCore(config, a);
        return { text: frameSummary(bundle, a.tick, a.out ? `完整 JSON 落在 ${a.out}` : '') };
      } catch (e) {
        return {
          text: `没跑成（${e?.code || 'error'}）：${String(e?.message || e).slice(-600)}`
            + `\n检查 cordis.patch.yml 里的 python（现在是 ${config.python}）和 workspace（${config.workspace}）。`,
        };
      }
    },
  };
}

/**
 * 激活插件。
 * @param ctx Cordis 上下文
 * @param rawConfig patch 里的 config
 */
export function apply(ctx, rawConfig) {
  const config = normalize(rawConfig);
  const tools = [createTool(config), createFrameTool(config)];
  ctx.inject?.(['tools'], (toolCtx) => {
    for (const tool of tools) {
      toolCtx.effect(() => toolCtx.tools.register(tool), `wangyun-tools: ${tool.name}`);
    }
  });
  ctx.logger?.info?.(`[wangyun-tools] 望云挂了上来：python=${config.python} workspace=${config.workspace}`);
}
