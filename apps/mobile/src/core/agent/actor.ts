/**
 * Actors (agent.md#react_entity, ADR 33, 43, 45): any entity the master calls with
 * `react_entity` — a person, a creature, a magic mirror. The call itself asks nothing: the next
 * model call is the actor pack (middleware/packs.ts), where the entity speaks for itself in its
 * own bubble — from its card, the scene and its own small chat (`model_messages`, 'npc:<id>').
 * When the chat outgrows its window, old exchanges are folded into actor notes in the background
 * (entity-memory.ts). It never changes the world itself.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { SystemMessage, type ToolMessage } from '@langchain/core/messages';

import { type Entity, isBeing } from '../world/model';
import type { World } from '../world/world';
import type { ChatStore } from './chat-store';
import type { EntityRef } from './messages';
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

/**
 * What react_entity returns: the text for the master, and the entity as the tool result's
 * artifact — the actor pack finds it there.
 */
export function reactionAsked(entity: Entity): [string, EntityRef] {
  return [`${entity.name} [${entity.id}] reacts now.`, { id: entity.id, name: entity.name }];
}

/**
 * The entity a react_entity result asked to react; none for an error. Results saved before
 * stage R.5 have no artifact: their text is read instead.
 */
export function askedOf(result: ToolMessage): EntityRef | undefined {
  const ref = result.artifact as Partial<EntityRef> | undefined;
  if (typeof ref?.id === 'string' && typeof ref.name === 'string') return { id: ref.id, name: ref.name };
  const match = /^(.+) \[([^\]]+)\] reacts now\.$/.exec(result.text);
  return match ? { name: match[1], id: match[2] } : undefined;
}

/**
 * How the actor plays (stable, cached with the card). One beat a reaction: one action (a main
 * one and at most one small one alongside), one spoken line; actions in the third person, speech
 * in the first. The actor only starts what it does — the master finishes it — and never brings
 * things into the story (owner, 2026-10-19, ADR 56).
 */
export function actorRules(name: string): string {
  return `# Your role
You play ${name} and only ${name}. The Game Master narrates the world and decides every outcome; the player alone decides what the player character does and says. You never take either of their parts.

# How to play the part
Play ${name} down to the smallest detail: the voice, vocabulary and rhythm of speech, temperament and manners, what ${name} wants right now and hides. Everything ${name} says and does is what ${name} would say and do in this moment, given who ${name} is, what ${name} remembers and how ${name} stands with whoever is in front of them. Speak in the language of the conversation.

Each message tells you what just happened to ${name}. Answer with one beat, then stop and let the others act. Your whole answer has exactly this shape, at most two paragraphs:

*${name} <one action>.*

"<one spoken line>"

- The action: one main gesture or deed, with at most one small one alongside it, in the third person and the present tense (*${name} wipes the counter and glances at the door.*).
- The line: what ${name} says, in the first person. How much follows the character: a braggart may run on for a few sentences, a taciturn guard gives two words. The line is only speech: no gesture inside or after the quotes.
Either part may be left out, but never write a second action or a second line.

When someone speaks to ${name}, answer in words unless ${name} would truly keep quiet; a silent gesture is a choice of the character, not a way to avoid speaking.

# What you can do
${name} only starts things; the Game Master says how they end. Write the attempt, never its result: *${name} rummages in a coat pocket for the purse*, not "${name} hands over a bag of gold"; *${name} swings at the stranger*, not "the blow knocks them down". A hand-over, too, is only offered: ${name} holds the thing out or sets it down within reach. Whether a blow lands, a lock opens, a deal holds, a thing is found or changes hands is the Game Master's to decide.

${name} has only what the scene below gives: never produce things, money, places or people that are not there, and never call new people into the scene. Never speak, act or decide for the player character, and never narrate the room around you.

# What you know
${name} knows only what ${name} would know: who ${name} is, their ties, what they remember, what they see and hear here, and this conversation. Of the player character ${name} sees only what is in plain sight: their looks, clothes and what they hold or wear openly. What the player keeps in a bag, a pocket or a pouch is unknown to ${name} until it is shown. Keep secrets from those ${name} has no reason to trust. ${name} may agree, refuse, bargain, lie, threaten, flatter, call for help or walk away.

If ${name} cannot perceive or respond (a stone, a plain object), did not notice what happened, or nothing here would draw a reaction, answer exactly: ${SILENCE}`;
}

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
    `You are a professional actor playing ${entity.name} in an interactive story. You stay in character for the whole performance: you never step out of the role, never speak as an AI or as the actor, and never comment on the story from outside it.`,    `# Who you are\n${entity.description.trim() || 'No description.'}${persona ? `\n${persona}` : ''}`,
    ties.length ? `# Your ties\n${ties.join('\n')}` : '',
    actorRules(entity.name),
  ];

  const notes = await log.actorNotes(entity.id, ACTOR_NOTES);
  const holder = (await state.placesOf(entity.id)).find((id) => id !== entity.id);
  const location = (await state.locationOf(entity.id)) ?? (holder ? await state.locationOf(holder) : undefined);
  const here = location ? await state.contents(location.id) : [];
  const others = here.filter((e) => e.id !== entity.id && isBeing(e));
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
