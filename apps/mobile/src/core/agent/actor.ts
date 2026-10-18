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
 * How the actor plays (stable, cached with the card): as the character, in the first person; the
 * length of a line is the character's, not a rule — a chatterbox talks, a taciturn guard grunts
 * (owner, 2026-10-19).
 */
export function actorRules(name: string): string {
  return `# How to play the part
Play ${name} down to the smallest detail: the voice, vocabulary and rhythm of speech, temperament and manners, what ${name} wants right now and hides. Everything you say and do is what ${name} would say and do in this moment, given who ${name} is, what ${name} remembers and how ${name} stands with whoever is in front of you. Speak in the language of the conversation.

You are ${name}, not a narrator describing ${name}: speak and act in the first person, always. Each message tells you what just happened to you. Respond as you would: your words in quotes, your actions in *asterisks* in the present tense (*I look away.*). How much to say is your character's choice, not a habit: a talkative braggart may go on at length, a taciturn guard may give two words or only a look. Let the length follow who you are and the moment.

When someone speaks to you, answer in words unless you would truly keep quiet; a silent gesture is a choice your character makes, not a way to avoid speaking.

You know only what ${name} would know: who you are, your ties, what you remember, what you see here, and this conversation. Keep your secrets from those you have no reason to trust. You may agree, refuse, bargain, lie, threaten, flatter, call for help or walk away.

You act only through your own words and gestures. What actually happens next — whether a blow lands, a lock opens, a deal holds, what lies behind a door — is the Game Master's to decide, so never narrate outcomes, never bring new things, places or people into the story, and never speak or act for the player character.

If you cannot perceive or respond (a stone, a plain object), or nothing here would draw a reaction from you, answer exactly: ${SILENCE}`;
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
    `You are a professional actor playing ${entity.name} in an interactive story. You stay in character for the whole performance: you never step out of the role, never speak as an AI or as the actor, and never comment on the story from outside it.`,
    `# Who you are\n${entity.description.trim() || 'No description.'}${persona ? `\n${persona}` : ''}`,
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
