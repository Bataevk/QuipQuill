import { AIMessage, type BaseMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { createMiddleware } from 'langchain';

import type { Entity } from '../../world/model';
import type { World } from '../../world/world';
import { type ActorDeps, actorChannel, actorSystemMessage, askedOf, isSilence, SILENCE } from '../actor';
import { estimateTokens, turnStart } from '../messages';

/** `additional_kwargs` key of an entity's own line written in the actor pack: `{ id, name }`. */
const ACTOR_LINE = 'qq_actor';
/** `additional_kwargs` key of a line that could not be written: the reason. */
const ACTOR_FAILED = 'qq_actor_failed';

/** The most of an entity's chat the actor pack sends; the background notes keep it near 4K. */
const ENTITY_CHAT_CAP_TOKENS = 8_000;

type Ref = { id: string; name: string };

/** The entity whose line this is (written in the actor pack), or undefined. */
export function actorOf(message: BaseMessage | undefined): Ref | undefined {
  if (!message || !AIMessage.isInstance(message)) return undefined;
  const ref = message.additional_kwargs?.[ACTOR_LINE] as Ref | undefined;
  return ref && typeof ref.id === 'string' ? ref : undefined;
}

/**
 * Which pack the next model call of the turn is in (ADR 43, 45), from the turn's messages alone
 * — so a retried turn picks up where it broke:
 * - **actor**: the master's last reply asked entities with react_entity, all its calls have
 *   results, and one of them has no line yet → that entity speaks, with its situation;
 * - **standard** otherwise: the master, with his usual prompt and tools (ADR 46: nothing is
 *   added after a line, so the cached prefix holds).
 */
export type Pack = { kind: 'standard' } | { kind: 'actor'; entity: Ref; situation: string };

export function packOf(messages: BaseMessage[]): Pack {
  const turn = messages.slice(Math.max(turnStart(messages), 0));
  const at = turn.findLastIndex((m) => AIMessage.isInstance(m) && !actorOf(m));
  if (at >= 0) {
    const asked = turn[at] as AIMessage;
    const after = turn.slice(at + 1);
    const results = new Map(after.flatMap((m) => (ToolMessage.isInstance(m) ? [[m.tool_call_id, m] as const] : [])));
    const done = (asked.tool_calls ?? []).every((call) => call.id && results.has(call.id));
    const spoke = after.flatMap((m) => actorOf(m) ?? []);
    for (const call of done ? (asked.tool_calls ?? []) : []) {
      const entity = call.name === 'react_entity' ? askedOf(results.get(call.id!)!.text) : undefined;
      if (!entity || spoke.some((s) => s.id === entity.id)) continue;
      return { kind: 'actor', entity, situation: String(call.args.situation ?? '') };
    }
  }
  return { kind: 'standard' };
}

/** A line as the master sees it: whose it is, so it is not taken for his own narration. */
function shownLine(message: BaseMessage): BaseMessage {
  const ref = actorOf(message);
  if (!ref) return message;
  const failed = message.additional_kwargs?.[ACTOR_FAILED];
  const content = failed
    ? `Error: ${ref.name} [${ref.id}] could not react right now (${String(failed)}). Narrate around it.`
    : message.text.trim()
      ? `${ref.name} [${ref.id}]: ${message.text.trim()}`
      : `${ref.name} [${ref.id}] does not react.`;
  return new AIMessage({ id: message.id, content });
}

export interface PacksOptions extends ActorDeps {
  world: World;
}

/**
 * Packs (ADR 43, 45; the idea from AI Samba, not its code). After react_entity the next call is
 * the **actor** pack: a small call of its own — the entity's card and the scene as the system
 * prompt, its own chat (`npc:<id>`) and the situation as messages, no tools — streamed in its
 * bubble. Then the middleware (only here) journals `entity.reacted`, adds the exchange to the
 * entity's chat and puts the line into the master's history as an AI message marked `qq_actor`.
 * Then the master goes on with his standard prompt and tools; he sees each line as
 * "Name [id]: …".
 */
export function packsMiddleware({ world, chats, model }: PacksOptions) {
  /** The entity's line: its model on its chat, then journal and chat — once per situation. */
  async function speak(entity: Entity, situation: string, signal?: AbortSignal): Promise<string> {
    const chatId = actorChannel(entity);
    const saved = await chats.load(chatId);
    // A retried turn may have got this far before: the reaction is journaled and in the chat.
    const [lastAsk, lastLine] = saved.slice(-2);
    if (lastAsk?.text === situation && lastLine && (await world.hasReacted(entity.id))) {
      return isSilence(lastLine.text) ? '' : lastLine.text;
    }
    let history = saved;
    while (history.length > 2 && estimateTokens(history) > ENTITY_CHAT_CAP_TOKENS) history = history.slice(2);
    const ask = new HumanMessage(situation);
    const answer = await model(entity).invoke([await actorSystemMessage(world, entity), ...history, ask], { signal });
    const raw = answer.text.trim();
    const silent = !raw || isSilence(raw);
    // The journal first: a reaction the player sees must count for the world's rules (ADR 34).
    const recorded = await world.recordReaction({ entity: entity.id, situation, reaction: silent ? SILENCE : raw });
    if (!recorded.ok) throw new Error(recorded.error);
    await chats.append(chatId, [ask, new AIMessage(silent ? SILENCE : raw)]);
    return silent ? '' : raw;
  }

  return createMiddleware({
    name: 'Packs',
    wrapModelCall: async (request, handler) => {
      const pack = packOf(request.messages);
      if (pack.kind === 'actor') {
        const signal = request.runtime?.signal;
        const tag = (extra: Record<string, unknown>) => ({ [ACTOR_LINE]: pack.entity, ...extra });
        try {
          const entity = await world.state.entity(pack.entity.id);
          if (!entity) throw new Error('it is gone');
          const text = await speak(entity, pack.situation, signal);
          return new AIMessage({ content: text, additional_kwargs: tag({}) });
        } catch (error) {
          if (signal?.aborted) throw error;
          return new AIMessage({ content: '', additional_kwargs: tag({ [ACTOR_FAILED]: (error as Error).message }) });
        }
      }
      return handler({ ...request, messages: request.messages.map(shownLine) });
    },
    afterModel: {
      canJumpTo: ['model'],
      // A line is not the end of the turn: the master speaks after it.
      hook: (state) => (actorOf(state.messages.at(-1)) ? { jumpTo: 'model' as const } : undefined),
    },
  });
}
