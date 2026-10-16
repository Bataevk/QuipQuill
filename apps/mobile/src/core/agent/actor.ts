/**
 * Actors (agent.md#react_entity, ADR 33, 43, 45): any entity the master calls with
 * `react_entity` — a person, a creature, a magic mirror. The call itself asks nothing: the next
 * model call is the actor pack (middleware/packs.ts), where the entity speaks for itself in its
 * own bubble — from its card, the scene and its own small chat (`model_messages`, 'npc:<id>').
 * When the chat outgrows its window, old exchanges are folded into actor notes in the background
 * (entity-memory.ts). It never changes the world itself.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { SystemMessage } from '@langchain/core/messages';

import type { Entity } from '../world/model';
import type { World } from '../world/world';
import type { ChatStore } from './chat-store';
import { brief } from './prompt';

export const SILENCE = '[silence]';

/** How many actor notes the actor sees. */
const ACTOR_NOTES = 12;

export interface ActorDeps {
  /**
   * The actor pack's model for one entity: in the app the queue model on channel 'npc:<id>',
   * purpose 'npc', with its name for the bubble (`queueModels().actorModel`).
   */
  model: (entity: Entity) => BaseChatModel;
  /** Where each entity's chat lives (`model_messages`, chat id 'npc:<id>'). */
  chats: ChatStore;
}

/** The entity's channel, chat id and `transcript` speaker: 'npc:<id>'. */
export const actorChannel = (entity: { id: string }) => `npc:${entity.id.replace(/^npc:/, '')}`;

/** What react_entity returns to the master; the actor pack finds the entity by it. */
export function reactionAsked(entity: Entity): string {
  return `${entity.name} [${entity.id}] reacts now.`;
}

/** The entity a react_entity result asked to react; none for an error. */
export function askedOf(result: string): { id: string; name: string } | undefined {
  const match = /^(.+) \[([^\]]+)\] reacts now\.$/.exec(result);
  return match ? { name: match[1], id: match[2] } : undefined;
}

export const ACTOR_RULES = `# How to react
- Each message tells what just happened, from your point of view. React as you would, in the first person: what you say, in quotes, and/or a short action in *asterisks*. One to three sentences.
- Know only what you would know: who you are, your ties, your notes, what you see here, and this conversation. Keep your secrets from strangers unless they give you a reason.
- You may agree, refuse, bargain, lie, threaten, call for help or walk away, but you cannot change the world: no new things, places or events beyond your own words and gestures. The Game Master decides what really happens.
- If you cannot perceive or care (a stone, a plain object), or nothing here calls for a reaction, answer exactly: ${SILENCE}`;

/**
 * The actor's system prompt: who it is (description, persona, ties) — stable, with a cache
 * breakpoint — then its notes, where it is now, what it carries, who is here.
 */
export async function actorSystemMessage(world: World, entity: Entity): Promise<SystemMessage> {
  const { state, log } = world;
  const persona = typeof entity.props.persona === 'string' ? entity.props.persona.trim() : '';
  const ties = (await state.ties(entity.id)).map(
    (t) => `- ${t.other.name}${t.description ? `: ${t.description}` : ''}`,
  );
  const stable = [
    `You are ${entity.name}, in an interactive story. Stay in character at all times.`,
    `# Who you are\n${entity.description.trim() || 'No description.'}${persona ? `\n${persona}` : ''}`,
    ties.length ? `# Your ties\n${ties.join('\n')}` : '',
    ACTOR_RULES,
  ];

  const notes = await log.actorNotes(entity.id, ACTOR_NOTES);
  const holder = (await state.placesOf(entity.id)).find((id) => id !== entity.id);
  const location = (await state.locationOf(entity.id)) ?? (holder ? await state.locationOf(holder) : undefined);
  const here = location ? await state.contents(location.id) : [];
  const others = here.filter((e) => e.id !== entity.id && (e.type === 'NPC' || e.type === 'CREATURE'));
  const carried = await state.inventory(entity.id);
  const player = await state.player();
  const playerHere = !!player && !!location && (await state.locationOf(player.id))?.id === location.id;
  const heldBy = holder && holder !== location?.id ? await state.entity(holder) : undefined;
  const scene = [
    notes.length ? `# What you remember\n${notes.map((n) => `- ${n.text}`).join('\n')}` : '',
    '# Now',
    location ? `You are in ${location.name}: ${brief(location.description, 300)}` : '',
    heldBy ? `${heldBy.type === 'PLAYER' ? 'The player character' : heldBy.name} carries you.` : '',
    others.length ? `Also here: ${others.map((e) => e.name).join(', ')}.` : '',
    carried.length ? `You carry: ${carried.map((e) => e.name).join(', ')}.` : '',
    player && playerHere
      ? `Facing you: the player character, ${player.name}${raceOf(player)} — ${brief(player.description, 300)}`
      : '',
  ];
  return new SystemMessage({
    content: [
      { type: 'text', text: stable.filter(Boolean).join('\n\n'), cache_control: { type: 'ephemeral' } },
      { type: 'text', text: scene.filter(Boolean).join('\n\n') },
    ],
  });
}

function raceOf(player: Entity): string {
  return typeof player.props.race === 'string' && player.props.race ? ` (${player.props.race})` : '';
}

export function isSilence(text: string): boolean {
  return text.replace(/[*_"'`.\s]/g, '').toLowerCase() === 'silence' || text.toLowerCase().startsWith(SILENCE);
}
