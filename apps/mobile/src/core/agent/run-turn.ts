import { AIMessage, type BaseMessage, ToolMessage } from '@langchain/core/messages';

import type { WorldEvent } from '../world/model';
import type { World } from '../world/world';
import { ChatSync, GM_CHAT, type ChatStore } from './chat-store';
import { actorChannel } from './actor';
import type { GmAgent } from './gm-agent';
import { actorOf } from './middleware/packs';
import { inputMessage, type InputSource, turnOf, turnStart } from './messages';
import type { TranscriptStore } from './transcript';
import { exclusive } from './turn-lock';

/** LangGraph steps per turn; middleware nodes count too. ToolLimit ends turns long before. */
const RECURSION_LIMIT = 150;

export interface TurnDeps {
  agent: GmAgent;
  world: World;
  chats: ChatStore;
  transcript: TranscriptStore;
  signal?: AbortSignal;
}

export interface TurnInput extends TurnDeps {
  /** What the player wrote, or a system command (not shown, not screened by the Guard). */
  input: string;
  source?: Exclude<InputSource, 'summary'>;
}

export interface TurnResult {
  turn: number;
  /** All the master's text of the turn, in order (the Guard's reply on reject); not entities' lines. */
  reply: string;
  /** This turn's messages in the master's chat, its input first. */
  messages: BaseMessage[];
  /** What changed in the world this turn. */
  events: WorldEvent[];
  /** The Guard rejected the input; the master was not called. */
  rejected: boolean;
  /** The turn ended at the tool call limit without a final answer. */
  cutShort: boolean;
}

/** The last turn already has its answer: there is nothing to retry. */
export class NothingToRetryError extends Error {
  constructor() {
    super('The last turn is complete; there is nothing to retry.');
    this.name = 'NothingToRetryError';
  }
}

/**
 * One turn: the master's history from `model_messages`, the input, the agent loop
 * (model ⇄ tools, with the middleware), and the history saved after every step, so a turn cut
 * short (network, app killed) keeps what already happened; BugGuard repairs its loose end.
 * The player's input goes to `transcript` at once, each piece of the master's text and each
 * entity's line (actor pack, under its own name) as soon as the model writes it, in the order
 * the player saw them.
 * One turn at a time per playthrough (TurnInProgressError).
 */
export function runTurn({ input, source = 'player', ...deps }: TurnInput): Promise<TurnResult> {
  return exclusive(deps.world, async () => {
    const turn = deps.world.beginTurn();
    const history = await deps.chats.load(GM_CHAT);
    if (source === 'player') await deps.transcript.append('player', input);
    return play(deps, turn, history, [...history, inputMessage(input, source, turn)]);
  });
}

/**
 * Finishes the last turn after a failure (network, app killed) without sending the input
 * again: the agent continues from the saved history, under the same turn number.
 */
export function retryTurn(deps: TurnDeps): Promise<TurnResult> {
  return exclusive(deps.world, async () => {
    const history = await deps.chats.load(GM_CHAT);
    if (!isUnfinished(history)) throw new NothingToRetryError();
    const start = turnStart(history);
    const known = turnOf(history[start]);
    const turn = known === undefined ? deps.world.beginTurn() : deps.world.continueTurn(known);
    return play(deps, turn, history, history);
  });
}

/** The master's last turn has no final answer (it failed or was cut off): retryTurn can finish it. */
export function isUnfinished(history: BaseMessage[]): boolean {
  const last = history.at(-1);
  if (turnStart(history) < 0 || !last) return false;
  // An entity's line is not the end: the master answers after it (packs, ADR 43).
  return !(AIMessage.isInstance(last) && !last.tool_calls?.length && !actorOf(last));
}

async function play(
  { agent, world, chats, transcript, signal }: TurnDeps,
  turn: number,
  saved: BaseMessage[],
  start: BaseMessage[],
): Promise<TurnResult> {
  const sync = new ChatSync(chats, GM_CHAT, saved);
  // AI texts already in the history were shown (and transcribed) before.
  const shown = new Set(start.filter((m) => AIMessage.isInstance(m)).map((m) => m.id));
  let messages: BaseMessage[] = start;
  const stream = await agent.stream(
    { messages: start },
    { streamMode: 'values', signal, recursionLimit: RECURSION_LIMIT },
  );
  for await (const state of stream) {
    messages = state.messages;
    await sync.save(messages);
    for (const message of messages.slice(Math.max(turnStart(messages), 0))) {
      if (!AIMessage.isInstance(message) || shown.has(message.id) || !message.text.trim()) continue;
      shown.add(message.id);
      const actor = actorOf(message);
      if (actor) {
        await transcript.append(actorChannel(actor), message.text.trim(), { name: actor.name });
        continue;
      }
      const rejected = message.response_metadata?.guard === 'reject';
      await transcript.append('gm', message.text.trim(), rejected ? { guard: 'reject' } : {});
    }
  }

  const own = messages.slice(Math.max(turnStart(messages), 0));
  const texts = own.filter((m) => AIMessage.isInstance(m) && m.text.trim() && !actorOf(m)) as AIMessage[];
  const lastMessage = own.at(-1);
  return {
    turn,
    reply: texts.map((m) => m.text.trim()).join('\n\n'),
    messages: own,
    events: await world.log.ofTurn(turn),
    rejected: texts.some((m) => m.response_metadata?.guard === 'reject'),
    cutShort:
      !!lastMessage && ToolMessage.isInstance(lastMessage) && lastMessage.additional_kwargs?.qq_blocked === true,
  };
}
