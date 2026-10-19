/**
 * Entity memory (ADR 45): each entity that reacts keeps a small chat of its own ('npc:<id>':
 * the master's situation → its line). When a chat outgrows ENTITY_CHAT_WINDOW_TOKENS, after the
 * turn and in the background, its oldest exchanges are folded into actor notes — one structured
 * call (`note_entity`) for all such entities at once — and dropped from the chat.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { type BaseMessage, HumanMessage, type SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import type { Entity, WorldEvent } from '../world/model';
import type { World } from '../world/world';
import { actorChannel } from '../protocol/channels';
import { ChatSync, type ChatStore } from './chat-store';
import { promptMessage, promptPart } from './cloud-prompt';
import { estimateTokens } from './messages';
import { brief } from './prompt';
import { invokeWithin, jsonSchemaFormat, parseJsonAnswer, strictObject } from './structured';
import { exclusive } from './turn-lock';

/** An entity's chat window: past it, the oldest exchanges are folded into notes. */
export const ENTITY_CHAT_WINDOW_TOKENS = 4_000;
/** What stays in the chat after folding (the latest exchanges). */
const KEEP_TOKENS = 2_000;
const NOTES_SHOWN = 12;


/** OpenAI-style `response_format` for the notes call. */
export const NOTE_ENTITY_RESPONSE_FORMAT = jsonSchemaFormat(
  'note_entity',
  strictObject({
    notes: { type: 'array', items: strictObject({ entity: { type: 'string' }, text: { type: 'string' } }) },
  }),
);

export const Notes = z.object({
  notes: z.array(z.object({ entity: z.string().trim().min(1), text: z.string().trim().min(1) })).default([]),
});

export interface EntityMemoryDeps {
  /** QueueChatModel with purpose 'summary' and NOTE_ENTITY_RESPONSE_FORMAT. */
  model: BaseChatModel;
  world: World;
  chats: ChatStore;
  signal?: AbortSignal;
  /** Explicit limit for the call (default 30 s); the next turn waits at most this long. */
  timeoutMs?: number;
}

export interface EntityMemoryResult {
  /** `note.added` events of the new actor notes. */
  noted: WorldEvent[];
  /** Entities whose chats were folded. */
  folded: string[];
  /** Notes that could not be written (an entity not in this batch, a refused note). */
  rejected: { entity: string; error: string }[];
}

interface Overflow {
  entity: Entity;
  chat: BaseMessage[];
  /** How many of the oldest messages are folded (whole exchanges). */
  fold: number;
}

/**
 * After a turn: folds the overflowing chats of the entities that reacted in it. Under the turn
 * lock, like the generator; nothing overflows → no call. A failed call folds nothing (the
 * actor pack caps what it sends), so the next turn tries again.
 */
export function noteEntities(deps: EntityMemoryDeps, turn: number): Promise<EntityMemoryResult> {
  return exclusive(deps.world, async () => {
    const result: EntityMemoryResult = { noted: [], folded: [], rejected: [] };
    const overflows = await overflowing(deps, turn);
    if (!overflows.length) return result;

    const answer = await invokeWithin(deps.model, await notesMessages(deps.world, overflows), {
      timeoutMs: deps.timeoutMs ?? 30_000,
      signal: deps.signal,
    });
    const { notes } = parseJsonAnswer(Notes, answer.text);

    await deps.world.actingAs('system', async () => {
      for (const note of notes) {
        const o = overflows.find((x) => x.entity.id === note.entity || x.entity.name === note.entity);
        if (!o) {
          result.rejected.push({ entity: note.entity, error: 'not one of the characters given' });
          continue;
        }
        const added = await deps.world.addActorNote(o.entity.id, note.text);
        if (added.ok) result.noted.push(added.value);
        else result.rejected.push({ entity: note.entity, error: added.error });
      }
    });
    for (const o of overflows) {
      await new ChatSync(deps.chats, actorChannel(o.entity), o.chat).save(o.chat.slice(o.fold));
      result.folded.push(o.entity.id);
    }
    return result;
  });
}

/** The entities of this turn's reactions whose chats are past the window, with what to fold. */
async function overflowing({ world, chats }: EntityMemoryDeps, turn: number): Promise<Overflow[]> {
  const ids = new Set<string>();
  for (const e of await world.log.ofTurn(turn)) if (e.kind === 'entity.reacted') ids.add(e.payload.entity);
  const found: Overflow[] = [];
  for (const id of ids) {
    const entity = await world.state.entity(id);
    if (!entity) continue;
    const chat = await chats.load(actorChannel(entity));
    if (estimateTokens(chat) <= ENTITY_CHAT_WINDOW_TOKENS) continue;
    let fold = 0;
    while (chat.length - fold > 2 && estimateTokens(chat.slice(fold)) > KEEP_TOKENS) fold += 2;
    if (fold > 0) found.push({ entity, chat, fold });
  }
  return found;
}

async function notesMessages(world: World, overflows: Overflow[]): Promise<[SystemMessage, HumanMessage]> {
  const blocks = await Promise.all(
    overflows.map(async ({ entity, chat, fold }) => {
      const notes = await world.log.actorNotes(entity.id, NOTES_SHOWN);
      const lines = chat
        .slice(0, fold)
        .map((m) => (m.getType() === 'human' ? `- Happened: ${m.text}` : `- ${entity.name}: ${m.text}`));
      return [
        `## ${entity.name} [${entity.id}]`,
        `Who: ${brief(entity.description, 400)}`,
        `Notes: ${notes.map((n) => n.text).join(' | ') || 'none'}`,
        'About to forget:',
        ...lines,
      ].join('\n');
    }),
  );
  return [
    promptMessage(promptPart('note_entity', {}, { type: 'ephemeral' })),
    new HumanMessage(blocks.join('\n\n')),
  ];
}
