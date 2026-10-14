import { AIMessage, type BaseMessage, HumanMessage } from '@langchain/core/messages';

import type { WorldEvent } from '../world/model';
import type { World } from '../world/world';
import { GM_CHAT, type ChatStore } from './chat-store';
import type { GmAgent } from './gm-agent';

export interface TurnInput {
  agent: GmAgent;
  world: World;
  chats: ChatStore;
  /** What the player wrote. */
  input: string;
  signal?: AbortSignal;
}

export interface TurnResult {
  turn: number;
  /** The master's final text for the player. */
  reply: string;
  /** This turn's messages as stored in the master's chat (player input first). */
  messages: BaseMessage[];
  /** What changed in the world this turn. */
  events: WorldEvent[];
}

/**
 * One player turn: the master's history from `model_messages`, the player's input, the agent
 * loop (model ⇄ tools), then the turn's messages appended to the history. World changes are
 * committed by the tools as they run; the history is saved only when the turn completes.
 */
export async function runTurn({ agent, world, chats, input, signal }: TurnInput): Promise<TurnResult> {
  const turn = world.beginTurn();
  const history = await chats.load(GM_CHAT);
  const result = await agent.invoke({ messages: [...history, new HumanMessage(input)] }, { signal });
  const messages = result.messages.slice(history.length);
  await chats.append(GM_CHAT, messages);
  const last = messages.findLast((message) => AIMessage.isInstance(message));
  return { turn, reply: last ? last.text : '', messages, events: await world.log.ofTurn(turn) };
}
