import { AIMessage, type BaseMessage, ToolMessage } from '@langchain/core/messages';

import type { WorldEvent } from '../world/model';
import type { World } from '../world/world';
import { ChatSync, GM_CHAT, type ChatStore } from './chat-store';
import type { GmAgent } from './gm-agent';
import { inputMessage, type InputSource, turnStart } from './messages';
import type { TranscriptStore } from './transcript';

/** LangGraph steps per turn; middleware nodes count too. ToolLimit ends turns long before. */
const RECURSION_LIMIT = 150;

export interface TurnInput {
  agent: GmAgent;
  world: World;
  chats: ChatStore;
  transcript: TranscriptStore;
  /** What the player wrote, or a system command (not shown, not screened by the Guard). */
  input: string;
  source?: Exclude<InputSource, 'summary'>;
  signal?: AbortSignal;
}

export interface TurnResult {
  turn: number;
  /** The master's final text for the player (the Guard's on reject). */
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

/**
 * One turn: the master's history from `model_messages`, the input, the agent loop
 * (model ⇄ tools, with the middleware), and the history saved after every step, so a turn cut
 * short (network, app killed) keeps what already happened; BugGuard repairs its loose end.
 * The player's input and the final reply go to `transcript`.
 */
export async function runTurn({
  agent,
  world,
  chats,
  transcript,
  input,
  source = 'player',
  signal,
}: TurnInput): Promise<TurnResult> {
  const turn = world.beginTurn();
  const history = await chats.load(GM_CHAT);
  if (source === 'player') await transcript.append('player', input);
  const sync = new ChatSync(chats, GM_CHAT, history);
  let messages: BaseMessage[] = [];
  const stream = await agent.stream(
    { messages: [...history, inputMessage(input, source)] },
    { streamMode: 'values', signal, recursionLimit: RECURSION_LIMIT },
  );
  for await (const state of stream) {
    messages = state.messages;
    await sync.save(messages);
  }

  const own = messages.slice(Math.max(turnStart(messages), 0));
  const answer = own.findLast((m) => AIMessage.isInstance(m) && m.text.trim()) as AIMessage | undefined;
  const rejected = answer?.response_metadata?.guard === 'reject';
  const reply = answer?.text.trim() ?? '';
  if (reply) await transcript.append('gm', reply, rejected ? { guard: 'reject' } : {});
  const lastMessage = own.at(-1);
  return {
    turn,
    reply,
    messages: own,
    events: await world.log.ofTurn(turn),
    rejected,
    cutShort: !!lastMessage && ToolMessage.isInstance(lastMessage) && lastMessage.additional_kwargs?.qq_blocked === true,
  };
}
