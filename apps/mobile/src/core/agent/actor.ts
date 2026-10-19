/**
 * Actors (agent.md#react_entity, ADR 33, 43, 45): any entity the master calls with
 * `react_entity` — a person, a creature, a magic mirror. The call itself asks nothing: the next
 * model call is the actor pack (middleware/packs.ts), where the entity speaks for itself in its
 * own bubble — from its card, the scene and its own small chat (`model_messages`, 'npc:<id>').
 * When the chat outgrows its window, old exchanges are folded into actor notes in the background
 * (entity-memory.ts). It never changes the world itself.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { SystemMessage, ToolMessage } from '@langchain/core/messages';

import { type Entity, isBeing } from '../world/model';
import type { World } from '../world/world';
import type { ChatStore } from './chat-store';
import { promptMessage, promptPart } from './cloud-prompt';
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
 * The actor's system prompt: the cloud's template `actor` (ADR 57) — how to play the part (ADR 53,
 * 56) — with who it is (description, persona, ties) as `CARD`, stable, with a cache breakpoint;
 * then its notes, where it is now, what it carries, who is here.
 */
export async function actorSystemMessage(world: World, entity: Entity): Promise<SystemMessage> {
  const { state, log } = world;
  const persona = typeof entity.props.persona === 'string' ? entity.props.persona.trim() : '';
  const ties = (await state.ties(entity.id)).map(
    (t) => `- ${t.other.name}${t.description ? `: ${t.description}` : ''}`,
  );
  const card = [
    `# Who you are\n${entity.description.trim() || 'No description.'}${persona ? `\n${persona}` : ''}`,
    ties.length ? `# Your ties\n${ties.join('\n')}` : '',
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
  return promptMessage(
    promptPart('actor', { NAME: entity.name, CARD: card.filter(Boolean).join('\n\n') }, { type: 'ephemeral' }),
    { type: 'text', text: scene.filter(Boolean).join('\n\n') },
  );
}

function raceOf(player: Entity): string {
  return typeof player.props.race === 'string' && player.props.race ? ` (${player.props.race})` : '';
}

export function isSilence(text: string): boolean {
  return text.replace(/[*_"'`.\s]/g, '').toLowerCase() === 'silence' || text.toLowerCase().startsWith(SILENCE);
}
