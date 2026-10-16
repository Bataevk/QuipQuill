/**
 * Actors (agent.md#react_entity, ADR 33, 43): any entity the master calls with `react_entity` —
 * a person, a creature, a magic mirror. The actor is one short call that answers the master,
 * not the player: what it feels and wants (its inner voice), or "[silence]". It has no chat of
 * its own; it remembers through its actor notes and its last reactions in the journal. It never
 * changes the world itself.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

import type { Entity } from '../world/model';
import type { World } from '../world/world';
import { brief } from './prompt';

export const SILENCE = '[silence]';

/** How many actor notes and own past reactions the actor sees. */
const ACTOR_NOTES = 12;
const PAST_REACTIONS = 4;

export interface ActorDeps {
  /** The inner voice: in the app the queue model with purpose 'npc' on the hidden channel 'actor'. */
  model: (entity: Entity) => BaseChatModel;
  /**
   * The actor pack (stage 6.2h): the entity's line on its own channel `npc:<id>`, with its name
   * for the bubble (`queueModels().actorLineModel`). Without it the master's model writes it.
   */
  lineModel?: (entity: Entity) => BaseChatModel;
  timeoutMs?: number;
}

/** Where the entity's own line streams and who it is in `transcript`: 'npc:<id>' (ADR 43). */
export const actorChannel = (entity: { id: string }) => `npc:${entity.id.replace(/^npc:/, '')}`;

/** What react_entity returns to the master for a reaction with words. */
export function voiceResult(entity: Entity, voice: string): string {
  return `Inner voice of ${entity.name} [${entity.id}] — for you only, the player has not heard it:\n${voice}`;
}

/** The entity a react_entity result gave an inner voice to; none for silence or an error. */
export function voiceOf(result: string): { id: string; name: string } | undefined {
  const match = /^Inner voice of (.+) \[([^\]]+)\] — for you only/.exec(result);
  return match ? { name: match[1], id: match[2] } : undefined;
}

export interface Reaction {
  entity: Entity;
  /** The inner voice for the master; empty for silence. */
  voice: string;
  silent: boolean;
}

export const ACTOR_RULES = `# How to answer
- You are not speaking to anyone. The Game Master asks how you take what just happened; answer them with your inner voice, in the first person: what you feel, what you think of it, and what you want to do or say now. One to three short sentences, like notes to yourself ("Wary. I want to know who sent them before I tell them anything.") — not your lines or gestures: the Game Master will play those.
- Be honest with the Game Master even when you would lie to others: if you mean to deceive, say so and say what the truth is.
- Know only what you would know: who you are, your ties, your notes, what you see here. Keep your secrets from strangers unless they give you a reason.
- You may want to agree, refuse, bargain, lie, threaten, call for help or walk away, but you cannot change the world. The Game Master decides what really happens.
- If you cannot perceive or care (a stone, a plain object), or nothing here calls for a reaction, answer exactly: ${SILENCE}`;

/**
 * The actor's system prompt: who it is (description, persona, ties) — stable, with a cache
 * breakpoint — then its notes and last reactions, where it is now, what it carries, who is here.
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
  const past = (await log.reactionsOf(entity.id, PAST_REACTIONS)).flatMap((e) =>
    e.kind === 'entity.reacted' && e.payload.reaction !== SILENCE
      ? [`- ${brief(e.payload.situation, 200)} → you felt: ${brief(e.payload.reaction, 300)}`]
      : [],
  );
  const holder = (await state.placesOf(entity.id)).find((id) => id !== entity.id);
  const location = (await state.locationOf(entity.id)) ?? (holder ? await state.locationOf(holder) : undefined);
  const here = location ? await state.contents(location.id) : [];
  const others = here.filter((e) => e.id !== entity.id && (e.type === 'NPC' || e.type === 'CREATURE'));
  const carried = await state.inventory(entity.id);
  const player = await state.player();
  const playerHere = !!player && !!location && (await state.locationOf(player.id))?.id === location.id;
  const heldBy = holder && holder !== location?.id ? await state.entity(holder) : undefined;
  const scene = [
    notes.length ? `# Your notes (what you decided before)\n${notes.map((n) => `- ${n.text}`).join('\n')}` : '',
    past.length ? `# Lately\n${past.join('\n')}` : '',
    '# Now',
    location ? `You are in ${location.name}: ${brief(location.description, 300)}` : '',
    heldBy ? `${heldBy.type === 'PLAYER' ? 'The player character' : heldBy.name} carries you.` : '',
    others.length ? `Also here: ${others.map((e) => e.name).join(', ')}.` : '',
    carried.length ? `You carry: ${carried.map((e) => e.name).join(', ')}.` : '',
    player && playerHere
      ? `The player character: ${player.name}${raceOf(player)} — ${brief(player.description, 300)}`
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

/** One reaction: the actor's model answers, `entity.reacted` goes to the journal. */
export async function reactEntity(
  world: World,
  deps: ActorDeps,
  entity: Entity,
  situation: string,
  signal?: AbortSignal,
): Promise<Reaction> {
  const timeout = AbortSignal.timeout(deps.timeoutMs ?? 60_000);
  const answer = await deps
    .model(entity)
    .invoke([await actorSystemMessage(world, entity), new HumanMessage(situation)], {
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
  const raw = answer.text.trim();
  const silent = !raw || isSilence(raw);
  const voice = silent ? '' : raw;
  // The journal: a reaction must count for the world's rules (ADR 34).
  const recorded = await world.recordReaction({ entity: entity.id, situation, reaction: silent ? SILENCE : voice });
  if (!recorded.ok) throw new Error(recorded.error);
  return { entity, voice, silent };
}

function isSilence(text: string): boolean {
  return text.replace(/[*_"'`.\s]/g, '').toLowerCase() === 'silence' || text.toLowerCase().startsWith(SILENCE);
}
