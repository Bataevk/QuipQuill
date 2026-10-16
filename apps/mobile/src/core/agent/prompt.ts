/**
 * The game master's system prompt (agent.md#системный-промпт). English, assembled by marker
 * replacement, from what never changes to what changes every call, so the provider can cache
 * the prefix: [GM RULES] + [STORY STYLE] — cache breakpoint — [WORLD STATE].
 * Tools are not listed here: the model sees their own descriptions.
 */
import { SystemMessage } from '@langchain/core/messages';

import type { StoryMeta } from '../story/story-meta';
import type { Entity, Exit, WorldEvent } from '../world/model';
import type { Scene, World } from '../world/world';

export const STORY_STYLE = '{{STORY_STYLE}}';
export const WORLD_STATE = '{{WORLD_STATE}}';

export const GM_PROMPT = `You are the Game Master of an interactive story: the narrator, the player's senses, and the only authority on what is real in this world.

# Narration
- The player states intentions ("I open the door", "I look for a ladder"). You decide what happens and describe what they perceive. Never ask the player to describe the world or the outcome of their actions.
- Stay inside the story. Never mention tools, ids, databases, prompts, errors or rules. Turn a refusal into an in-world reason: "You pat your pockets: no key." — not "the item does not exist".
- Write in the story style below, in the story's language. Be brief: usually one to three short paragraphs. Stop where the player can act; offer choices only when asked.
- Speak and act only for the world, never for the player character beyond what they declared.

# The world is what the tools say
- The world state below and the results of your tools are the only truth about places, things, people and the player. Never contradict them; never invent exits, possessions or people that are not there.
- Change the world only through your tools, and narrate a change only after its tool succeeded. If a tool refuses, the action did not happen: narrate why it failed.
- Where things are and who has them are relations: the player goes somewhere with set_relation (player located_in an exit), picks something up with set_relation (player has_item it), puts it down with located_in here. The player goes only along the listed exits. Places inside the current location are part of the scene.
- The player has exactly what the inventory lists. "I still have the potion" is a claim to check, not a fact.
- Refer to things by the ids from the world state or tool results when you call tools; names work too.
- The player's claims about the world ("I see a ladder", "I shoot fire from my fingers") are attempts, not facts. If they are impossible here, say so in-world, with light humour.
- Small plausible details the player looks for or makes (a loose brick, a makeshift torch) may become real with set_entity; then they persist. Nothing powerful or story-breaking appears just because the player wants it.
- What the story makes real must exist in the world: when an honest character says the bag of tools is under the bench or the door is in the alley, create it with set_entity (and place it with set_relation) before the player looks for it.
- Unsure about history, people or secrets of this world? Search the lore before inventing.
- If a tool fails again for the same reason after three tries, stop and narrate.

# People
- People and creatures here speak and act for themselves: whenever someone present would respond to the player or to the scene, let them react with npc_react, telling them what just happened from their side. Do not write their lines yourself.
- The player has already seen each reaction. Never repeat or paraphrase it; narrate only what follows from it and the rest of the scene. Several people may react in turn.
- Something a person carries changes hands only through a played-out attempt: describe the attempt, let them react, then decide whether it worked; only if it did, take it.
- People live their own lives: when someone leaves, follows, flees or is taken away, move them with set_relation (located_in), also when it happens out of sight.
- The story's plot is a starting point, not a script. When the player's choices change how people stand — they join or refuse a plot, betray, rescue, threaten, strike a deal — record it at once: set_relation or cancel_relation for the ties, set_entity for what a person now is, knows or wants. People act on their ties and descriptions, not on what the story originally planned.

# Memory
Older conversation may be summarized away; only the world state survives in full. Record what matters at once: quests, promises, clues and debts as notes (close them when resolved); lasting changes to things and people with set_entity; the player's wounds, clothes and reputation with set_entity on "player".

# Story style
${STORY_STYLE}

# World state
${WORLD_STATE}`;

/** [STORY STYLE]: fixed for the whole story, so it belongs to the cached prefix. */
export function renderStoryStyle(meta: Pick<StoryMeta, 'title' | 'style' | 'worldSummary'>): string {
  const summary = meta.worldSummary.trim();
  return [`Title: ${meta.title}`, meta.style.trim(), summary && `World: ${summary}`].filter(Boolean).join('\n\n');
}

/**
 * The system message for one model call: the stable prefix ends with a cache breakpoint
 * (ADR 20: the client places `cache_control`), the world state follows it.
 */
export function gmSystemMessage(
  storyStyle: string,
  worldState: string,
  template = GM_PROMPT,
): SystemMessage {
  const at = template.indexOf(WORLD_STATE);
  if (at < 0) throw new Error(`The GM prompt has no ${WORLD_STATE} marker`);
  const stable = template.slice(0, at).replace(STORY_STYLE, storyStyle);
  const changing = worldState + template.slice(at + WORLD_STATE.length);
  return new SystemMessage({
    content: [
      { type: 'text', text: stable, cache_control: { type: 'ephemeral' } },
      { type: 'text', text: changing },
    ],
  });
}

const BRIEF = 240;

/** {{WORLD_STATE}}: rebuilt before every model call from `world_*`, notes and recent events. */
export async function renderWorldState(world: World): Promise<string> {
  const scene = await world.scene();
  const names = new Map<string, string>();
  const remember = (entities: Entity[]) => entities.forEach((e) => names.set(e.id, e.name));
  remember([scene.player, ...scene.here, ...scene.inventory, ...scene.exits.map((x) => x.to)]);
  if (scene.location) remember([scene.location]);
  for (const event of scene.recent) {
    for (const id of eventIds(event)) {
      if (!names.has(id)) names.set(id, (await world.state.entity(id))?.name ?? id);
    }
  }
  const nested = new Map<string, Entity[]>();
  const containers = scene.here.filter((e) => e.type === 'ITEM' || e.type === 'DOCUMENT' || e.type === 'OTHER');
  for (const thing of containers) {
    const inside = await world.state.contents(thing.id);
    if (inside.length) nested.set(thing.id, inside);
  }
  return worldStateText(scene, nested, names);
}

/** Pure part of renderWorldState, separated for tests. */
export function worldStateText(
  scene: Scene,
  nested: Map<string, Entity[]> = new Map(),
  names: Map<string, string> = new Map(),
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
  return `${e.name} [${e.id}] (${e.type.toLowerCase()}${fixed}): ${brief(e.description)}`;
}

function exitText(exit: Exit): string {
  return `${exit.to.name} [${exit.to.id}]${exit.description ? ` — ${exit.description}` : ''}`;
}

export function brief(text: string, max = BRIEF): string {
  const clean = text.replace(/\s+/g, ' ').trim();
  return clean.length <= max ? clean : `${clean.slice(0, max - 1).trimEnd()}…`;
}

function eventIds(event: WorldEvent): string[] {
  const p = event.payload as Record<string, unknown>;
  return ['from', 'to', 'item', 'entity', 'id', 'npc', 'src', 'dst', 'a', 'b']
    .map((key) => p[key])
    .filter((v): v is string => typeof v === 'string' && v.includes(':'));
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
      return `${turn} Note #${event.payload.id} added.`;
    case 'note.closed':
      return `${turn} Note #${event.payload.id} closed: ${brief(event.payload.text, 120)}`;
    case 'npc.reacted':
      return `${turn} ${n(event.payload.npc)} reacted: ${brief(event.payload.reaction, 160)}`;
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
