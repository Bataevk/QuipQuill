import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, type BaseMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { createMiddleware, tool } from 'langchain';
import { z } from 'zod';

import type { Entity } from '../../world/model';
import type { World } from '../../world/world';
import { voiceOf } from '../actor';
import { answeredIds, turnStart } from '../messages';
import { brief } from '../prompt';

/** `additional_kwargs` key of an entity's own line written in the actor pack: `{ id, name }`. */
const ACTOR_LINE = 'qq_actor';

export const ACTOR_NOTE = 'actor_note';

type Ref = { id: string; name: string };

/** The entity whose line this is (written in the actor pack), or undefined. */
export function actorOf(message: BaseMessage | undefined): Ref | undefined {
  if (!message || !AIMessage.isInstance(message)) return undefined;
  const ref = message.additional_kwargs?.[ACTOR_LINE] as Ref | undefined;
  return ref && typeof ref.id === 'string' ? ref : undefined;
}

/**
 * Which pack the next model call of the turn is in (ADR 43), from the turn's messages alone —
 * so a retried turn picks up where it broke:
 * - **actor**: the master's last reply asked entities with react_entity, all its calls have
 *   results, and one of the inner voices has no line yet → that entity speaks;
 * - **master**: entities have spoken this turn → the master again, with actor_note and the
 *   instruction to note and answer, which stays right after the last line for every call of the
 *   pack (so after the notes the model still knows it owes the player an answer);
 * - **standard** otherwise.
 */
export type Pack =
  | { kind: 'standard' }
  | { kind: 'actor'; entity: Ref }
  | { kind: 'master'; spoke: Ref[] };

export function packOf(messages: BaseMessage[]): Pack {
  const start = Math.max(turnStart(messages), 0);
  const turn = messages.slice(start);
  const lines = turn.flatMap((m) => actorOf(m) ?? []);
  const at = turn.findLastIndex((m) => AIMessage.isInstance(m) && !actorOf(m));
  if (at >= 0) {
    const asked = turn[at] as AIMessage;
    const after = turn.slice(at + 1);
    const answered = answeredIds(after);
    const done = (asked.tool_calls ?? []).every((call) => call.id && answered.has(call.id));
    const voices = after.flatMap((m) =>
      ToolMessage.isInstance(m) && m.name === 'react_entity' ? (voiceOf(m.text) ?? []) : [],
    );
    const spoke = after.flatMap((m) => actorOf(m) ?? []);
    const next = voices.find((v) => !spoke.some((s) => s.id === v.id));
    if (done && next) return { kind: 'actor', entity: next };
  }
  if (lines.length) {
    const spoke = [...new Map(lines.map((l) => [l.id, l])).values()];
    return { kind: 'master', spoke };
  }
  return { kind: 'standard' };
}

export function actorInstruction(entity: Entity): string {
  return [
    `[Actor] Now speak and act as ${entity.name} [${entity.id}] — only them. Who they are: ${brief(entity.description, 500)}`,
    'Write what they say (in quotes) and do (in *asterisks*) right now, true to their inner voice above: one to three sentences, in their own manner. If they mean to lie, lie as they would.',
    'Do not narrate anyone else, the outcome or the scene, and do not call tools: the player sees this as their own words.',
  ].join('\n');
}

export function masterInstruction(spoke: Ref[]): string {
  const names = spoke.map((r) => r.name).join(' and ');
  return [
    `[Game Master] ${names} had their say; the player saw it.`,
    `Briefly note with ${ACTOR_NOTE} what they decided or now want (a promise, a refusal, a plan, a lie they told): they remember only what is noted. If how they stand with someone changed, record that tie with set_relation too.`,
    'If they honestly spoke of something in this world (a door, a bag), make it real with set_entity / set_relation now; a lie stays a lie — the inner voice tells which.',
    'Then answer the player: only what follows and the rest of the scene. Never repeat their words.',
  ].join('\n');
}

/** A line as the master sees it in later calls: whose it is, so it is not taken for his own narration. */
function shownLine(message: BaseMessage): BaseMessage {
  const ref = actorOf(message);
  if (!ref) return message;
  return new AIMessage({ id: message.id, content: `${ref.name} [${ref.id}]: ${message.text}` });
}

export interface PacksOptions {
  world: World;
  /** The entity's line on its own channel; without it the master's model writes it. */
  lineModel?: (entity: Entity) => BaseChatModel;
}

/**
 * Packs (ADR 43, stage 6.2h; the idea from AI Samba, not its code): after a react_entity with an
 * inner voice the next call is the **actor** pack — the entity's line, streamed in its own
 * bubble and kept in the master's history as an AI message marked with `qq_actor`; then the
 * **master** pack — the standard one plus the dynamic tool actor_note and, right after the
 * last line, the instruction to note the entity's decisions and answer the player. A new turn is the
 * standard pack again. The packs keep the master's system prompt and tools (actor_note added),
 * so the cached prefix holds; their instructions go after the history and are not saved.
 */
export function packsMiddleware({ world, lineModel }: PacksOptions) {
  const actorNote = tool(
    async ({ text, who }: { text: string; who?: string }) => {
      try {
        const reacted = await world.reactedThisTurn();
        if (!reacted.length) return 'Error: no one reacted this turn; there is nobody to note for.';
        let entity = reacted.at(-1)!;
        if (who) {
          const found = world.state.resolve(who, reacted);
          if (found.kind !== 'found') {
            return `Error: "${who}" did not react this turn. Choose one of: ${reacted.map((e) => e.name).join(', ')}.`;
          }
          entity = found.entity;
        }
        const noted = await world.addActorNote(entity.id, text);
        return noted.ok ? `Noted for ${entity.name}.` : `Error: ${noted.error}`;
      } catch (error) {
        return `Error: ${ACTOR_NOTE} failed unexpectedly (${(error as Error).message}). Go on without it.`;
      }
    },
    {
      name: ACTOR_NOTE,
      description:
        'A short note for whoever just reacted: what they decided, promised, refused or now want, in their ' +
        'own terms ("I refused to tell the stranger about the vault; I want them gone"). They remember only ' +
        'their notes and their last reactions. Several may be added.',
      schema: z.object({
        text: z.string().describe('The note, in the first person of the one who reacted'),
        who: z.string().optional().describe('Who it is for, if several reacted this turn (default: the last)'),
      }),
    },
  );
  const isNote = (t: unknown) => (t as { name?: string }).name === ACTOR_NOTE;

  return createMiddleware({
    name: 'Packs',
    tools: [actorNote],
    wrapModelCall: async (request, handler) => {
      const pack = packOf(request.messages);
      const messages = request.messages.map(shownLine);
      const tools = request.tools.filter((t) => !isNote(t));
      if (pack.kind === 'actor') {
        const entity = (await world.state.entity(pack.entity.id)) ?? ({ ...pack.entity, description: '' } as Entity);
        const response = await handler({
          ...request,
          model: lineModel ? lineModel(entity) : request.model,
          tools,
          messages: [...messages, new HumanMessage(actorInstruction(entity))],
        });
        if (!AIMessage.isInstance(response)) return response;
        // A line, never actions: tool calls written in the actor pack are dropped.
        return new AIMessage({
          id: response.id,
          content: response.content,
          additional_kwargs: { ...response.additional_kwargs, [ACTOR_LINE]: pack.entity },
          usage_metadata: response.usage_metadata,
          response_metadata: response.response_metadata,
        });
      }
      if (pack.kind === 'master') {
        const at = request.messages.findLastIndex((m) => actorOf(m));
        return handler({
          ...request,
          tools: [...tools, actorNote],
          messages: [
            ...messages.slice(0, at + 1),
            new HumanMessage(masterInstruction(pack.spoke)),
            ...messages.slice(at + 1),
          ],
        });
      }
      return handler({ ...request, tools, messages });
    },
    afterModel: {
      canJumpTo: ['model'],
      // A line is not the end of the turn: the master speaks after it.
      hook: (state) => (actorOf(state.messages.at(-1)) ? { jumpTo: 'model' as const } : undefined),
    },
  });
}
