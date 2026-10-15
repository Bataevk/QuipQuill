import type { ServerEvent } from './messages';

/** What one action (a turn, a proposal) cost, from the `usage` of each `done` (stage 6.1). */
export interface ActionUsage {
  calls: number;
  input: number;
  output: number;
  cacheRead: number;
  costUsd: number;
  /** Per channel kind: 'gm', 'npc', 'guard', 'system' (summaries, adaptation). */
  byChannel: Record<string, { calls: number; costUsd: number }>;
}

export const emptyUsage = (): ActionUsage => ({ calls: 0, input: 0, output: 0, cacheRead: 0, costUsd: 0, byChannel: {} });

export function addUsage(total: ActionUsage, part: ActionUsage): ActionUsage {
  const byChannel = { ...total.byChannel };
  for (const [channel, value] of Object.entries(part.byChannel)) {
    const prev = byChannel[channel] ?? { calls: 0, costUsd: 0 };
    byChannel[channel] = { calls: prev.calls + value.calls, costUsd: prev.costUsd + value.costUsd };
  }
  return {
    calls: total.calls + part.calls,
    input: total.input + part.input,
    output: total.output + part.output,
    cacheRead: total.cacheRead + part.cacheRead,
    costUsd: total.costUsd + part.costUsd,
    byChannel,
  };
}

/**
 * Sums the cost of the generations an action started. A request counts once, and only if it
 * began during the action (its `status`): after `resume` the server repeats `done` of
 * generations finished up to 15 minutes ago. Cancelled or failed generations report no usage
 * (backlog 2.3).
 */
export class UsageMeter {
  private readonly started = new Set<string>();
  private readonly counted = new Set<string>();
  private usage = emptyUsage();

  apply(event: ServerEvent): void {
    if (!('requestId' in event) || !event.requestId) return;
    if (event.type === 'status') this.started.add(event.requestId);
    if (event.type !== 'done' || !this.started.has(event.requestId) || this.counted.has(event.requestId)) return;
    this.counted.add(event.requestId);
    const { usage } = event;
    const channel = event.channel?.startsWith('npc:') ? 'npc' : (event.channel ?? 'gm');
    const prev = this.usage.byChannel[channel] ?? { calls: 0, costUsd: 0 };
    this.usage = {
      calls: this.usage.calls + 1,
      input: this.usage.input + usage.input,
      output: this.usage.output + usage.output,
      cacheRead: this.usage.cacheRead + usage.cacheRead,
      costUsd: this.usage.costUsd + usage.costUsd,
      byChannel: { ...this.usage.byChannel, [channel]: { calls: prev.calls + 1, costUsd: prev.costUsd + usage.costUsd } },
    };
  }

  get total(): ActionUsage {
    return this.usage;
  }
}

/** "$0.0023 · 4 calls · 9.1K in (3.2K cached) · 420 out" */
export function formatUsage(usage: ActionUsage): string {
  const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}K` : `${n}`);
  const parts = Object.entries(usage.byChannel)
    .map(([channel, value]) => `${channel} ${value.calls}`)
    .join(', ');
  return (
    `$${usage.costUsd.toFixed(4)} · ${usage.calls} call${usage.calls === 1 ? '' : 's'}${parts ? ` (${parts})` : ''}` +
    ` · ${k(usage.input)} in (${k(usage.cacheRead)} cached) · ${k(usage.output)} out`
  );
}
