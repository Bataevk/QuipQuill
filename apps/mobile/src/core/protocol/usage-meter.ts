import { GM_CHANNEL, isActorChannel } from './channels';
import type { ServerEvent } from './messages';

/** What one action (a turn, a proposal) cost, from the `usage` of each `done` (stage 6.1). */
export interface ChannelUsage {
  calls: number;
  costUsd: number;
  input: number;
  cacheRead: number;
  /** Prompt tokens written to the provider's cache (ADR 49: to see what each caller caches). */
  cacheWrite: number;
}

export interface ActionUsage {
  calls: number;
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  costUsd: number;
  /** Per channel kind: 'gm', 'npc', 'guard', 'generator', 'system' (summaries, adaptation, memory). */
  byChannel: Record<string, ChannelUsage>;
}

export const emptyUsage = (): ActionUsage => ({
  calls: 0,
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  costUsd: 0,
  byChannel: {},
});

const emptyChannel = (): ChannelUsage => ({ calls: 0, costUsd: 0, input: 0, cacheRead: 0, cacheWrite: 0 });

function addChannel(a: ChannelUsage, b: ChannelUsage): ChannelUsage {
  return {
    calls: a.calls + b.calls,
    costUsd: a.costUsd + b.costUsd,
    input: a.input + b.input,
    cacheRead: a.cacheRead + b.cacheRead,
    cacheWrite: a.cacheWrite + b.cacheWrite,
  };
}

export function addUsage(total: ActionUsage, part: ActionUsage): ActionUsage {
  const byChannel = { ...total.byChannel };
  for (const [channel, value] of Object.entries(part.byChannel)) {
    byChannel[channel] = addChannel(byChannel[channel] ?? emptyChannel(), value);
  }
  return {
    calls: total.calls + part.calls,
    input: total.input + part.input,
    output: total.output + part.output,
    cacheRead: total.cacheRead + part.cacheRead,
    cacheWrite: total.cacheWrite + part.cacheWrite,
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
    const channel = isActorChannel(event.channel) ? 'npc' : (event.channel ?? GM_CHANNEL);
    const call: ChannelUsage = {
      calls: 1,
      costUsd: usage.costUsd,
      input: usage.input,
      cacheRead: usage.cacheRead,
      cacheWrite: usage.cacheWrite ?? 0,
    };
    this.usage = addUsage(this.usage, {
      ...call,
      output: usage.output,
      byChannel: { [channel]: call },
    });
  }

  get total(): ActionUsage {
    return this.usage;
  }
}

const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}K` : `${n}`);
const cached = (u: { cacheRead: number; cacheWrite: number }) =>
  `${k(u.cacheRead)} cached${u.cacheWrite ? `, ${k(u.cacheWrite)} written` : ''}`;

/** "$0.0023 · 4 calls (gm 2, guard 1, npc 1) · 9.1K in (3.2K cached, 1.0K written) · 420 out" */
export function formatUsage(usage: ActionUsage): string {
  const parts = Object.entries(usage.byChannel)
    .map(([channel, value]) => `${channel} ${value.calls}`)
    .join(', ');
  return (
    `$${usage.costUsd.toFixed(4)} · ${usage.calls} call${usage.calls === 1 ? '' : 's'}${parts ? ` (${parts})` : ''}` +
    ` · ${k(usage.input)} in (${cached(usage)}) · ${k(usage.output)} out`
  );
}

/** Per channel, for measurements: "gm 3: 40.1K in (30.2K cached, 5.1K written) $0.0021". */
export function formatChannels(usage: ActionUsage): string[] {
  return Object.entries(usage.byChannel).map(
    ([channel, u]) => `${channel} ${u.calls}: ${k(u.input)} in (${cached(u)}) $${u.costUsd.toFixed(4)}`,
  );
}
