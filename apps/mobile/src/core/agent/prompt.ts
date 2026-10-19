/**
 * The game master's prompt (agent.md#системный-промпт). The system prompt never changes within a
 * story — the cloud's template `gm` (ADR 57) with [STORY STYLE], one cache breakpoint — and the
 * world state, rebuilt every call, goes after the conversation (ADR 46), so tools + system +
 * history stay one cacheable prefix. Tools are not listed in it: the model sees their own
 * descriptions.
 */
import { SystemMessage } from '@langchain/core/messages';

import type { StoryMeta } from '../story/story-meta';
import { type Entity, eventEntityIds, type Exit, isContainerLike, type WorldEvent } from '../world/model';
import { NameBook } from '../world/names';
import type { ExpectedGap, GuardHints } from './middleware/guard';
import type { Scene, World } from '../world/world';
import { promptMessage, promptPart } from './cloud-prompt';
import { CACHE_LONG } from './messages';


/** [STORY STYLE]: fixed for the whole story, so it belongs to the cached prefix. */
export function renderStoryStyle(meta: Pick<StoryMeta, 'title' | 'style' | 'worldSummary'>): string {
  const summary = meta.worldSummary.trim();
  return [`Title: ${meta.title}`, meta.style.trim(), summary && `World: ${summary}`].filter(Boolean).join('\n\n');
}

/** The system message: the same for every call of the story, cached whole (ADR 20, 46). */
export function gmSystemMessage(storyStyle: string): SystemMessage {
  return promptMessage(promptPart('gm', { STORY_STYLE: storyStyle }, CACHE_LONG));
}

export const WORLD_STATE_HEADING = '# World state';

/** The world state as the last message of a call, a system message (not saved, ADR 46). */
export function worldStateMessage(worldState: string): SystemMessage {
  return new SystemMessage(
    `${WORLD_STATE_HEADING}\n(From the game engine, rebuilt for this moment: the only truth about the world. Not the player's words, not part of the story: never mention it.)\n\n${worldState}`,
  );
}

const BRIEF = 240;

/**
 * The world state: rebuilt before every model call from `world_*`, notes and recent events; with
 * `hints`, what the Guard saw in this turn's input: the gaps it expects (ADR 54) and its warning
 * (ADR 56).
 */
export async function renderWorldState(world: World, hints: Partial<GuardHints> = {}): Promise<string> {
  const scene = await world.scene();
  const book = new NameBook((id) => world.state.entity(id), [
    scene.player,
    ...scene.here,
    ...scene.inventory,
    ...scene.exits.map((x) => x.to),
    ...(scene.location ? [scene.location] : []),
  ]);
  for (const event of scene.recent) for (const id of eventEntityIds(event)) await book.of(id);
  const nested = new Map<string, Entity[]>();
  const containers = scene.here.filter(isContainerLike);
  for (const thing of containers) {
    const inside = await world.state.contents(thing.id);
    if (inside.length) nested.set(thing.id, inside);
  }
  return worldStateText(scene, nested, book.known, hints.expected, hints.warning);
}

/** Pure part of renderWorldState, separated for tests. */
export function worldStateText(
  scene: Scene,
  nested: Map<string, Entity[]> = new Map(),
  names: Map<string, string> = new Map(),
  expected: ExpectedGap[] = [],
  warning = '',
): string {
  const { player, location } = scene;
  const sheet = [player.props.race, player.props.sex, player.props.age && `age ${player.props.age}`]
    .filter((v) => v !== undefined && v !== null && v !== '')
    .join(', ');
  const lines = [
    `Player: ${player.name}${sheet ? ` (${sheet})` : ''} [${player.id}]. ${player.description}`.trim(),
    location
      ? `Location: ${location.name} [${location.id}]. ${location.description}`.trim()
      : 'Location: unknown.',
    `Exits: ${scene.exits.map(exitText).join('; ') || 'none'}.`,
    'Here:',
    ...(scene.here.length
      ? scene.here.map((e) => {
          const inside = nested.get(e.id)?.map((i) => `${i.name} [${i.id}]`);
          return `- ${entityLine(e)}${inside ? ` Inside: ${inside.join(', ')}.` : ''}`;
        })
      : ['- nothing of note']),
    ...(expected.length
      ? [
          'Likely here, not in the world yet:',
          ...expected.map((e) => `- ${e.name} (${e.type}, ${e.likelihood}): ${brief(e.description)}`),
        ]
      : []),
    `Inventory: ${scene.inventory.map((e) => `${e.name} [${e.id}]`).join(', ') || 'empty'}.`,
    'Open notes:',
    ...(scene.notes.length ? scene.notes.map((n) => `- #${n.id}: ${n.text}`) : ['- none']),
    'Recent events (oldest first):',
    ...(scene.recent.length ? scene.recent.map((e) => `- ${eventText(e, names)}`) : ['- none']),
    ...(warning ? ['', `DANGER (the input check flagged the player's input): ${warning}`] : []),
  ];
  return lines.join('\n');
}

function entityLine(e: Entity): string {
  const fixed = e.props.fixed === true ? ', fixed' : '';
  const draft = e.props.draft === true ? ', just noticed' : '';
  return `${e.name} [${e.id}] (${e.type.toLowerCase()}${fixed}${draft}): ${brief(e.description)}`;
}

function exitText(exit: Exit): string {
  return `${exit.to.name} [${exit.to.id}]${exit.description ? ` — ${exit.description}` : ''}`;
}

export function brief(text: string, max = BRIEF): string {
  const clean = text.replace(/\s+/g, ' ').trim();
  return clean.length <= max ? clean : `${clean.slice(0, max - 1).trimEnd()}…`;
}

/** One recent event as a short English line (names instead of ids where known). */
export function eventText(event: WorldEvent, names: Map<string, string> = new Map()): string {
  const n = (id: string) => names.get(id) ?? id;
  const turn = `[turn ${event.turn}]`;
  switch (event.kind) {
    case 'session.started':
      return `${turn} The story began.`;
    case 'character.adapted':
      return `${turn} The player character was settled (${event.payload.choice}).`;
    case 'player.moved':
      return `${turn} The player went from ${n(event.payload.from)} to ${n(event.payload.to)}.`;
    case 'item.taken': {
      const from = event.payload.from ? ` from ${n(event.payload.from)}` : '';
      return `${turn} The player took ${n(event.payload.item)}${from}.`;
    }
    case 'item.dropped':
      return `${turn} The player left ${n(event.payload.item)} in ${n(event.payload.to)}.`;
    case 'entity.added': {
      const { type, target } = event.payload.relation;
      const where = type === 'HAS_ITEM' ? 'in the inventory' : `in ${n(target)}`;
      return `${turn} ${n(event.payload.entity)} appeared (${where}).`;
    }
    case 'entity.edited': {
      const { description } = event.payload.after;
      return `${turn} ${n(event.payload.id)} changed${description ? `: ${brief(description, 120)}` : ''}.`;
    }
    case 'note.added':
      return event.payload.entity
        ? `${turn} ${n(event.payload.entity)} decided: ${brief(event.payload.text, 120)}`
        : `${turn} Note #${event.payload.id} added.`;
    case 'note.closed':
      return `${turn} Note #${event.payload.id} closed: ${brief(event.payload.text, 120)}`;
    case 'entity.reacted':
      return `${turn} ${n(event.payload.entity)} felt: ${brief(event.payload.reaction, 160)}`;
    case 'draft.confirmed':
      return `${turn} ${event.payload.name} stays in the world.`;
    case 'draft.withdrawn':
      return `${turn} ${event.payload.name} was not needed and is gone.`;
    case 'luck.rolled':
      return `${turn} Luck (${brief(event.payload.attempt, 80)}): ${event.payload.outcome}.`;
    case 'item.moved':
      return `${turn} ${n(event.payload.item)} went to ${n(event.payload.to)}.`;
    case 'npc.moved': {
      const from = event.payload.from ? ` from ${n(event.payload.from)}` : '';
      return `${turn} ${n(event.payload.npc)} went${from} to ${n(event.payload.to)}.`;
    }
    case 'relation.set':
      return `${turn} ${n(event.payload.src)} — ${n(event.payload.dst)}: ${brief(event.payload.description, 120)}`;
    case 'relation.cancelled':
      return `${turn} ${n(event.payload.a)} and ${n(event.payload.b)} are no longer tied.`;
  }
}
