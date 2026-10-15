/**
 * NPC reactions (agent.md#npc_react, ADR 33): an NPC is not an agent but a chat of its own
 * (`model_messages` with chat_id 'npc:<id>') that the master calls with `npc_react`. The NPC
 * answers in words and/or a short action, or "[silence]"; it never changes the world itself.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';

import type { Entity } from '../world/model';
import type { World } from '../world/world';
import { ChatSync, type ChatStore } from './chat-store';
import { compactHistory } from './middleware/context';
import { brief } from './prompt';
import type { TranscriptStore } from './transcript';

export const SILENCE = '[silence]';

export const NPC_TRIMMED_NOTE = '[You remember only the latest part of this conversation.]';

/** NPC chats are small: kept by trimming (no summary call), well under any price step. */
const NPC_CONTEXT = { triggerTokens: 6_000, keepTokens: 3_000, maxTokens: 12_000, trimmedNote: NPC_TRIMMED_NOTE };

export interface NpcDeps {
  /**
   * The model for one NPC: in the app the queue model with channel 'npc:<id>', purpose 'npc'
   * and meta.speaker (see `npcQueueModel`), so the UI streams the NPC's bubble at once.
   */
  model: (npc: Entity) => BaseChatModel;
  chats: ChatStore;
  transcript: TranscriptStore;
  timeoutMs?: number;
}

export interface NpcReaction {
  npc: Entity;
  /** The NPC's words and/or deeds; empty for silence. */
  text: string;
  silent: boolean;
}

export const npcChatId = (npc: Entity) => `npc:${npc.id.replace(/^npc:/, '')}`;

export const NPC_RULES = `# How to react
- Each message tells what just happened, from your point of view. React as you would, in the first person: what you say, in quotes, and/or a short action in *asterisks*. One to three sentences.
- Know only what you would know: who you are, your ties, what you see here, and this conversation. Keep your secrets from strangers unless they give you a reason.
- You may agree, refuse, bargain, lie, threaten, call for help or walk away, but you cannot change the world: no new things, places or events beyond your own words and gestures. The Game Master decides what really happens.
- If you are not a living, thinking being, or nothing here calls for a reaction, answer exactly: ${SILENCE}`;

/**
 * The NPC's system prompt: who it is (description, persona, ties in the graph) — stable, with
 * a cache breakpoint — then where it is now, what it carries, who else is here.
 */
export async function npcSystemMessage(world: World, npc: Entity): Promise<SystemMessage> {
  const { state } = world;
  const persona = typeof npc.props.persona === 'string' ? npc.props.persona.trim() : '';
  const ties = (await state.ties(npc.id)).map(
    (t) => `- ${t.other.name}${t.description ? `: ${t.description}` : ''}`,
  );
  const stable = [
    `You are ${npc.name}, a character in an interactive story. Stay in character at all times.`,
    `# Who you are\n${npc.description.trim() || 'No description.'}${persona ? `\n${persona}` : ''}`,
    ties.length ? `# Your ties\n${ties.join('\n')}` : '',
    NPC_RULES,
  ];

  const location = await state.locationOf(npc.id);
  const here = location ? await state.contents(location.id) : [];
  const others = here.filter((e) => e.id !== npc.id && (e.type === 'NPC' || e.type === 'CREATURE'));
  const carried = await state.inventory(npc.id);
  const player = await state.player();
  const playerHere = !!player && !!location && (await state.locationOf(player.id))?.id === location.id;
  const scene = [
    '# Now',
    location ? `You are in ${location.name}: ${brief(location.description, 300)}` : '',
    others.length ? `Also here: ${others.map((e) => e.name).join(', ')}.` : '',
    `You carry: ${carried.map((e) => e.name).join(', ') || 'nothing of note'}.`,
    player && playerHere
      ? `Facing you: the player character, ${player.name}${raceOf(player)} — ${brief(player.description, 300)}`
      : '',
  ];
  return new SystemMessage({
    content: [
      { type: 'text', text: stable.filter(Boolean).join('\n\n'), cache_control: { type: 'ephemeral' } },
      { type: 'text', text: scene.filter(Boolean).join('\n') },
    ],
  });
}

function raceOf(player: Entity): string {
  return typeof player.props.race === 'string' && player.props.race ? ` (${player.props.race})` : '';
}

/**
 * One reaction: the situation joins the NPC's chat, its model answers; then `npc.reacted` goes
 * to the journal, the exchange to the NPC's chat, the line to the transcript (the player sees
 * it).
 */
export async function reactNpc(
  world: World,
  deps: NpcDeps,
  npc: Entity,
  situation: string,
  signal?: AbortSignal,
): Promise<NpcReaction> {
  const chatId = npcChatId(npc);
  const saved = await deps.chats.load(chatId);
  const history = (await compactHistory([...saved, new HumanMessage(situation)], NPC_CONTEXT)) ?? [
    ...saved,
    new HumanMessage(situation),
  ];
  const timeout = AbortSignal.timeout(deps.timeoutMs ?? 60_000);
  const answer = await deps
    .model(npc)
    .invoke([await npcSystemMessage(world, npc), ...history], {
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
  const raw = answer.text.trim();
  const silent = !raw || isSilence(raw);
  const text = silent ? '' : raw;

  // The journal first: a reaction the player sees must count for the world's rules (ADR 34).
  const recorded = await world.recordNpcReaction({ npc: npc.id, situation, reaction: silent ? SILENCE : text });
  if (!recorded.ok) throw new Error(recorded.error);
  await new ChatSync(deps.chats, chatId, saved).save([...history, new AIMessage(silent ? SILENCE : text)]);
  if (!silent) await deps.transcript.append(chatId, text, { name: npc.name });
  return { npc, text, silent };
}

function isSilence(text: string): boolean {
  return text.replace(/[*_"'`.\s]/g, '').toLowerCase() === 'silence' || text.toLowerCase().startsWith(SILENCE);
}
