import type { ServerEvent } from './messages';
import { StreamText } from './stream-text';

/** A bubble being streamed right now: the text of one generation so far. */
export interface LiveBubble {
  requestId: string;
  /** 'gm' or 'npc:<id>'. */
  channel: string;
  /** NPC name (from `meta.speaker`, or set later); undefined for the master. */
  speaker: string | undefined;
  text: string;
}

/** Channels drawn as bubbles: the master and NPCs (not the guard, summaries or adaptation). */
export const isStoryChannel = (channel: string | undefined) =>
  channel === 'gm' || (channel?.startsWith('npc:') ?? false);

/**
 * The live part of the game screen while one action runs (stage 5.2): a bubble per generation
 * of a story channel, built from the event stream by the protocol rules (StreamText: deltas in
 * order, `replay` replaces). Only generations that stream during the action get a bubble: after
 * `resume` the server repeats `done` / `error` of generations finished up to 15 minutes ago
 * (protocol.md#буфер-активной-генерации-и-resume) — those are already in the transcript.
 */
export class LiveFeed {
  private readonly streams = new Map<string, { stream: StreamText; channel: string; speaker?: string }>();
  private open = false;

  /** An action starts: accept events, empty feed. */
  begin(): void {
    this.streams.clear();
    this.open = true;
  }

  /** The action ended (its text is in the transcript now): drop everything, accept nothing. */
  end(): void {
    this.streams.clear();
    this.open = false;
  }

  /**
   * Applies one event. Returns 'new' when a bubble appeared (look up its speaker), 'changed'
   * when the visible text changed, undefined otherwise.
   */
  apply(event: ServerEvent): 'new' | 'changed' | undefined {
    if (!this.open || !('requestId' in event) || !event.requestId || !isStoryChannel(event.channel)) return undefined;
    let entry = this.streams.get(event.requestId);
    let created = false;
    if (!entry) {
      if (event.type !== 'delta' && event.type !== 'replay') return undefined; // late done/error, status
      const speaker = typeof event.meta?.speaker === 'string' ? event.meta.speaker : undefined;
      entry = { stream: new StreamText(event.requestId), channel: event.channel ?? 'gm', speaker };
      this.streams.set(event.requestId, entry);
      created = true;
    }
    const changed = entry.stream.apply(event);
    if (created) return 'new';
    return changed ? 'changed' : undefined;
  }

  speakerKnown(requestId: string): boolean {
    return this.streams.get(requestId)?.speaker !== undefined;
  }

  setSpeaker(requestId: string, speaker: string | undefined): void {
    const entry = this.streams.get(requestId);
    if (entry) entry.speaker = speaker;
  }

  channelOf(requestId: string): string | undefined {
    return this.streams.get(requestId)?.channel;
  }

  /** Bubbles with text, in the order their generations started. */
  bubbles(): LiveBubble[] {
    return [...this.streams.entries()]
      .map(([requestId, s]) => ({ requestId, channel: s.channel, speaker: s.speaker, text: s.stream.text }))
      .filter((bubble) => bubble.text.trim());
  }
}
