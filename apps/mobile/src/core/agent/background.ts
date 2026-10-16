import type { World } from '../world/world';
import { released } from './turn-lock';

/**
 * Work after a turn, in the background (stage 6.2i): the world generator and entity memory.
 * A job starts once the turn that asked for it has released the turn lock, one job at a time;
 * the next action waits for it (`settle`), closing the game aborts it (`cancel`). A job's own
 * failures are its business; whatever escapes goes to the developer's log, never to the player.
 */
export class BackgroundWork {
  private current: { done: Promise<void>; controller: AbortController } | undefined;
  private readonly around: (work: () => Promise<void>) => Promise<void>;

  /** `around` wraps each job, e.g. to meter what it costs. */
  constructor(options: { around?: (work: () => Promise<void>) => Promise<void> } = {}) {
    this.around = options.around ?? ((work) => work());
  }

  start(world: World, job: (signal: AbortSignal) => Promise<void>): void {
    const controller = new AbortController();
    const previous = this.current?.done ?? Promise.resolve();
    const done = previous
      .then(() => released(world))
      .then(() => this.around(() => job(controller.signal)))
      .catch((error: unknown) => {
        console.warn(`[background] ${error instanceof Error ? error.message : String(error)}`);
      });
    this.current = { done, controller };
  }

  /** Resolves when the running job (if any) has finished, failed or been aborted. */
  async settle(): Promise<void> {
    const running = this.current;
    if (!running) return;
    await running.done;
    if (this.current === running) this.current = undefined;
  }

  cancel(): void {
    this.current?.controller.abort();
  }
}
