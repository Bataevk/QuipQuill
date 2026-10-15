import { describe, expect, it } from 'vitest';

import type { ServerEvent } from './messages';
import { addUsage, emptyUsage, formatUsage, UsageMeter } from './usage-meter';

const status = (requestId: string, channel: string): ServerEvent => ({
  v: 1,
  type: 'status',
  status: 'thinking',
  requestId,
  sessionId: 's',
  channel,
});
const done = (requestId: string, channel: string, costUsd: number, input = 1000, cacheRead = 0): ServerEvent => ({
  v: 1,
  type: 'done',
  requestId,
  sessionId: 's',
  channel,
  message: { role: 'assistant', content: 'x' },
  finishReason: 'stop',
  usage: { input, output: 50, cacheRead, costUsd },
});

describe('UsageMeter', () => {
  it('sums the generations started during the action, by channel', () => {
    const meter = new UsageMeter();
    for (const [id, channel] of [['g', 'gm'], ['n', 'npc:mags'], ['q', 'guard'], ['g2', 'gm']] as const) {
      meter.apply(status(id, channel));
    }
    meter.apply(done('q', 'guard', 0.0003));
    meter.apply(done('g', 'gm', 0.001, 8000, 3000));
    meter.apply(done('n', 'npc:mags', 0.0004));
    meter.apply(done('g2', 'gm', 0.0009, 8500, 8000));
    expect(meter.total).toMatchObject({
      calls: 4,
      input: 18_500,
      output: 200,
      cacheRead: 11_000,
      byChannel: { guard: { calls: 1 }, gm: { calls: 2 }, npc: { calls: 1 } },
    });
    expect(meter.total.costUsd).toBeCloseTo(0.0026, 6);
    expect(formatUsage(meter.total)).toBe('$0.0026 · 4 calls (guard 1, gm 2, npc 1) · 18.5K in (11.0K cached) · 200 out');
  });

  it('ignores done repeated after resume, for old requests and twice for the same one', () => {
    const meter = new UsageMeter();
    meter.apply(done('old', 'gm', 0.5)); // finished before the action
    meter.apply(status('g', 'gm'));
    meter.apply(done('g', 'gm', 0.001));
    meter.apply(done('g', 'gm', 0.001)); // repeated after resume
    expect(meter.total).toMatchObject({ calls: 1, costUsd: 0.001 });
  });

  it('adds actions up', () => {
    const a = new UsageMeter();
    a.apply(status('g', 'gm'));
    a.apply(done('g', 'gm', 0.001));
    const total = addUsage(addUsage(emptyUsage(), a.total), a.total);
    expect(total).toMatchObject({ calls: 2, byChannel: { gm: { calls: 2 } } });
    expect(total.costUsd).toBeCloseTo(0.002, 6);
  });
});
