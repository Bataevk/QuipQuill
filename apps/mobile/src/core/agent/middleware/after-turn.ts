import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, type BaseMessage } from '@langchain/core/messages';
import { createMiddleware } from 'langchain';

import type { StoryMeta } from '../../story/story-meta';
import type { World } from '../../world/world';
import type { BackgroundWork } from '../background';
import type { ChatStore } from '../chat-store';
import { noteEntities } from '../entity-memory';
import { generate, type GeneratorResult } from '../generator';
import { sourceOf, turnOf, turnStart } from '../messages';
import { actorOf } from './packs';

/** The limit of each background call: the next action waits for it at most this long. */
export const AFTER_TURN_TIMEOUT_MS = 30_000;

export interface AfterTurnOptions {
  world: World;
  meta: Pick<StoryMeta, 'title' | 'style' | 'worldSummary'>;
  chats: ChatStore;
  work: BackgroundWork;
  /** The world generator (ADR 42); without it there is none. */
  generatorModel?: BaseChatModel;
  /** Entity memory, `note_entity` (ADR 45); without it there is none. */
  memoryModel?: BaseChatModel;
  timeoutMs?: number;
  /** What the generator did, or why it failed (dev diagnostics). */
  onGenerated?: (result: GeneratorResult | { error: string }) => void;
}

/**
 * AfterTurn (afterAgent, stage 6.2i, ADR 43): when the master has answered a player's turn,
 * starts the background work — the generator brings the world in line with the turn, then the
 * overflowing entity chats are folded into notes. The player does not wait: it runs once the
 * turn releases its lock, and the next action waits for it. Turns the Guard rejected and
 * system turns (the opening) have nothing to read.
 */
export function afterTurnMiddleware(options: AfterTurnOptions) {
  return createMiddleware({
    name: 'AfterTurn',
    afterAgent: (state) => {
      // The afterAgent state type leaves out the built-in messages, though they are there.
      const { messages } = state as unknown as { messages: BaseMessage[] };
      const turn = turnToSettle(messages);
      if (turn) startAfterTurn(options, turn);
      return;
    },
  });
}

/** A player's turn as the background work reads it. */
export interface SettledTurn {
  turn: number;
  input: string;
  /** All the master's text of the turn (entities' lines left out). */
  reply: string;
}

/** The last turn of the history if it needs background work: a player's, not rejected by the Guard. */
export function turnToSettle(messages: BaseMessage[]): SettledTurn | undefined {
  const start = turnStart(messages);
  const input = messages[start];
  const turn = input ? turnOf(input) : undefined;
  if (!input || turn === undefined || sourceOf(input) !== 'player') return undefined;
  const said = messages.slice(start).filter((m) => AIMessage.isInstance(m) && !actorOf(m)) as AIMessage[];
  if (said.some((m) => m.response_metadata?.guard === 'reject')) return undefined;
  const reply = said.flatMap((m) => (m.text.trim() ? [m.text.trim()] : [])).join('\n\n');
  return { turn, input: input.text, reply };
}

/**
 * Queues the background work of one turn: the generator, then entity memory; when both are
 * through (not aborted), the turn is marked done, so a game reopened after the app was killed
 * runs only what it missed (ADR 48).
 */
export function startAfterTurn(options: AfterTurnOptions, { turn, input, reply }: SettledTurn): void {
  const { world, meta, chats, work, generatorModel, memoryModel, onGenerated } = options;
  const timeoutMs = options.timeoutMs ?? AFTER_TURN_TIMEOUT_MS;
  if (!generatorModel && !memoryModel) return;
  work.start(world, async (signal) => {
    if (generatorModel) {
      try {
        const result = await generate({ model: generatorModel, world, meta, signal, timeoutMs }, { turn, input, reply });
        for (const r of result.rejected) console.warn(`[generator] turn ${turn}: ${r.change}: ${r.error}`);
        onGenerated?.(result);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.warn(`[generator] turn ${turn} failed: ${message}`);
        onGenerated?.({ error: message });
      }
    }
    if (memoryModel && !signal.aborted) {
      try {
        const noted = await noteEntities({ model: memoryModel, world, chats, signal, timeoutMs }, turn);
        for (const r of noted.rejected) console.warn(`[memory] turn ${turn}: ${r.entity}: ${r.error}`);
      } catch (error) {
        console.warn(`[memory] turn ${turn} failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    if (!signal.aborted) await world.markAfterTurnDone(turn);
  });
}
