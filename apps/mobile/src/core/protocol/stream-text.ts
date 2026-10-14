import type { ServerEvent } from './messages';

/**
 * The text the UI shows for one generation, following the protocol rules: deltas are applied
 * in `seq` order and duplicates dropped; `replay` (after reconnecting) replaces the text;
 * `done` carries the final text. Events of other requests are ignored.
 */
export class StreamText {
  readonly requestId: string;
  private value = '';
  private lastSeq = -1;
  private finished = false;

  constructor(requestId: string) {
    this.requestId = requestId;
  }

  get text(): string {
    return this.value;
  }

  get done(): boolean {
    return this.finished;
  }

  /** Returns true if the visible text changed. */
  apply(event: ServerEvent): boolean {
    if (!('requestId' in event) || event.requestId !== this.requestId || this.finished) return false;
    const before = this.value;
    switch (event.type) {
      case 'delta':
        if (event.seq <= this.lastSeq) return false; // duplicate after a reconnect
        this.lastSeq = event.seq;
        this.value += event.text;
        break;
      case 'replay':
        // Everything streamed so far. Live deltas that follow have higher seq numbers than
        // anything the replay covers, so `lastSeq` stays as is.
        this.value = event.text;
        break;
      case 'done':
        this.finished = true;
        this.value = event.message.content ?? this.value;
        break;
      case 'error':
        this.finished = true;
        break;
      default:
        return false;
    }
    return this.value !== before;
  }
}
