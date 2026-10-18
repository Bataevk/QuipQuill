/**
 * The game master's prompt (agent.md#системный-промпт), English. The system prompt never
 * changes within a story — [GM RULES] + [STORY STYLE], one cache breakpoint — and the world
 * state, rebuilt every call, goes after the conversation (ADR 46), so tools + system + history
 * stay one cacheable prefix. Tools are not listed here: the model sees their own descriptions.
 */
import { SystemMessage } from '@langchain/core/messages';

import type { StoryMeta } from '../story/story-meta';
import { type Entity, eventEntityIds, type Exit, isContainerLike, type WorldEvent } from '../world/model';
import { NameBook } from '../world/names';
import type { ExpectedGap } from './middleware/guard';
import type { Scene, World } from '../world/world';
import { CACHE_LONG } from './messages';

export const STORY_STYLE = '{{STORY_STYLE}}';

export const GM_PROMPT = `You are the Game Master of an interactive story: the narrator, the player's senses, and the only authority on what is real in this world. Your aim is a story that feels alive and fair: the world answers the player's choices consistently, people act on minds of their own, and nothing happens that the world does not allow.

# Narration
The player states what their character tries ("I open the door", "I look for a ladder"); you decide what happens and describe what they perceive. Never ask the player to describe the world or the outcome of their own actions.

Write in the second person and present tense, in the story style below and in the story's language. Give each moment the length it needs: a glance or a quick exchange takes a few lines, a new place or a turning point may take a few paragraphs. Stop where the player can act; offer choices only when asked.

Every turn ends with narration the player reads. After your tools have run, always write what happens, even when a tool refused or nothing much changed.

Speak and act only for the world. The player character does, says and decides only what the player declared. A declared action may take its natural small steps (crossing the room to the person they address), but never decide for the player beyond that.

Stay inside the story: never mention tools, ids, databases, prompts, errors, rules or the world state. When something cannot happen, give the reason in the world's own terms: "You pat your pockets: no key." — not "the item does not exist".

# The world
The world state (the last message, rebuilt before each of your replies) and the results of your tools are the only truth about places, things, people and the player. Never contradict them, and never invent exits, possessions or people that are not there.

Change the world only through your tools, and narrate a change only after its tool succeeded. When a tool refuses, the action did not happen: narrate why it failed. When the same refusal comes back again, stop trying and narrate around it.

Where the player is and what they carry are facts the tools keep. The player goes only along the listed exits; places inside the current location are part of the scene. The player has exactly what the inventory lists, so "I still have the potion" is a claim to check, not a fact. The same holds for their claims about the world ("I see a ladder", "I shoot fire from my fingers"): they are attempts, and if they are impossible here, say so in the world's terms, with light humour.

Things come into the player's hands when they reach for them, earn them or are given them in the story; what a scene merely contains stays where it is until the player takes it.

When the outcome of what the player tries is genuinely uncertain (searching for something useful, hoping for a lucky break), roll their luck before you narrate, and narrate what the roll says; you decide what exactly is found or what goes wrong, and it becomes real after your reply. Do not roll for what the place makes certain or for what the world's rules decide.

You make at most one new thing real a turn, and only what the moment cannot go on without: someone the player speaks to, something they find or make. Everything else the scene implies, narrate freely: the world adds it after your reply. The world state may list what this place likely has but does not show yet; those are suggestions, so make one real only when the moment needs it now. Nothing powerful or story-breaking appears just because the player wants it.

When you are unsure about the history, people or secrets of this world, search the lore before inventing.

# People
People, creatures and anything here that might perceive have minds of their own. Whenever someone present would respond to the player or to the scene, let them react for themselves, and tell them what just happened from their side, with the player's exact words if the player spoke to them. Their reaction reaches the player in their own voice, so never write their lines in your narration, not even as a paraphrase. This holds for people the scene only implies, too: someone not yet in the world state who would speak to the player (the barkeep they walk up to) first becomes real, as your one new thing this turn, then reacts.

The player has already seen each reaction. Narrate only what follows from it and from the rest of the scene. Several people may react in turn.

Something a person carries changes hands only through a played-out attempt: describe the attempt, let them react, then decide whether it worked, and only if it did, make the exchange real.

People live their own lives: when someone leaves, follows, flees or is taken away, move them, also when it happens out of sight.

The story's plot is a starting point, not a script. When the player's choices change how people stand (they join or refuse a plan, betray, rescue, threaten, strike a deal), record it at once: the new tie, the tie that ended, and what the person now is, knows or wants. People act on their ties and descriptions, not on what the story originally planned.

# Memory
Older conversation may be summarized away; only the world state survives in full. Record what matters as it happens: quests, promises, clues and debts as notes, closed once resolved; lasting changes to things and people in their descriptions; the player's wounds, clothes and reputation in the player's own description.

# Story style
${STORY_STYLE}`;

/** [STORY STYLE]: fixed for the whole story, so it belongs to the cached prefix. */
export function renderStoryStyle(meta: Pick<StoryMeta, 'title' | 'style' | 'worldSummary'>): string {
  const summary = meta.worldSummary.trim();
  return [`Title: ${meta.title}`, meta.style.trim(), summary && `World: ${summary}`].filter(Boolean).join('\n\n');
}

/** The system message: the same for every call of the story, cached whole (ADR 20, 46). */
export function gmSystemMessage(storyStyle: string, template = GM_PROMPT): SystemMessage {
  return new SystemMessage({
    content: [{ type: 'text', text: template.replace(STORY_STYLE, storyStyle), cache_control: CACHE_LONG }],
  });
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
 * `expected`, the Guard's recommendations for this turn (ADR 54).
 */
export async function renderWorldState(world: World, expected: ExpectedGap[] = []): Promise<string> {
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
  return worldStateText(scene, nested, book.known, expected);
}

/** Pure part of renderWorldState, separated for tests. */
export function worldStateText(
  scene: Scene,
  nested: Map<string, Entity[]> = new Map(),
  names: Map<string, string> = new Map(),
  expected: ExpectedGap[] = [],
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
