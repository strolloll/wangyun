/**
 * 装之前先在 node 里空跑一遍：import 插件、注册工具、真调一次 python。
 * 用法：node smoke.mjs
 */
import { apply } from './index.js';

const registered = new Map();
const ctx = {
  logger: { info: (s) => console.log('[info]', s), warn: (s) => console.warn('[warn]', s) },
  inject: (deps, fn) => {
    const toolCtx = {
      effect: (fn2) => fn2(),
      tools: { register: (tool) => { registered.set(tool.name, tool); return () => {}; } },
    };
    fn(toolCtx);
  },
};

apply(ctx, { workspace: 'D:\\望云', timeoutMs: 60000 });
for (const name of ['wy_observe', 'wy_frame']) {
  if (!registered.has(name)) {
    console.error('工具没注册上：', name);
    process.exit(1);
  }
}
console.log('工具：', [...registered.keys()].join('、'));

const args = { n: 32, modules: 4, ticks: 60, probe: 20, need: 2 };
const overview = await registered.get('wy_observe').execute(args);
console.log('--- wy_observe ---');
console.log(overview.text);

const frame = await registered.get('wy_frame').execute({ ...args, tick: 40 });
console.log('--- wy_frame（tick=40）---');
console.log(frame.text);

const detailed = await registered.get('wy_frame').execute({ ...args, tick: 40, members: true });
console.log('--- wy_frame（members=true）---');
console.log(detailed.text);
