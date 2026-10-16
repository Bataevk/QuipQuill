import type { World } from '../world/world';

/** Another turn (or start-of-session step) of this playthrough is still running. */
export class TurnInProgressError extends Error {
  constructor() {
    super('A turn is already in progress.');
    this.name = 'TurnInProgressError';
  }
}

const busy = new WeakSet<World>();
const waiting = new WeakMap<World, (() => void)[]>();

/**
 * One turn at a time per playthrough: two at once would interleave their world changes and
 * overwrite each other's saved history (a double tap, a retry while the turn still runs).
 * The second caller fails at once instead of waiting.
 */
export async function exclusive<T>(world: World, fn: () => Promise<T>): Promise<T> {
  if (busy.has(world)) throw new TurnInProgressError();
  busy.add(world);
  try {
    return await fn();
  } finally {
    busy.delete(world);
    const resolvers = waiting.get(world) ?? [];
    waiting.delete(world);
    resolvers.forEach((resolve) => resolve());
  }
}

/** Resolves once no turn holds the lock of this playthrough (at once if none does). */
export function released(world: World): Promise<void> {
  if (!busy.has(world)) return Promise.resolve();
  return new Promise((resolve) => waiting.set(world, [...(waiting.get(world) ?? []), resolve]));
}
