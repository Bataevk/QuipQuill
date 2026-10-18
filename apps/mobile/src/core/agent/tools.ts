/**
 * The game master's tools (agent.md#инструменты-мастера): thin wrappers over `World`
 * commands (ADR 29). A broken rule or any failure comes back to the model as text starting with
 * "Error:", never as an exception. Tool descriptions are part of the contract with the model,
 * as much as the prompt is.
 */
import type { StructuredToolInterface } from '@langchain/core/tools';
import { tool } from 'langchain';
import { z } from 'zod';

import { ADDABLE_TYPES, type Entity, isCarrier, lowerTypes, type Outcome, type WorldEvent } from '../world/model';
import { NameBook } from '../world/names';
import type { World } from '../world/world';
import { DEFAULT_LUCK } from '../story/story-meta';
import { type ActorDeps, reactionAsked } from './actor';
import { LIKELIHOODS, type Likelihood, type LuckOutcome, rollLuck, STAKES, type Stakes } from './luck';
import { brief } from './prompt';

export interface GmToolDeps {
  world: World;
  /** Query vector for search_lore, with the story's `meta.embed_model` (CloudEmbeddings). */
  embed?: (text: string) => Promise<Float32Array>;
  /** The actor pack's model; without it nothing can react. */
  actor?: ActorDeps;
  /** The story's luck, 0..1 (`meta.luck`, ADR 47); default DEFAULT_LUCK. */
  luck?: number;
  /** Chance for roll_luck, [0, 1); default Math.random. */
  random?: () => number;
}

export const GM_TOOL_NAMES = [
  'inspect',
  'search_lore',
  'set_entity',
  'set_relation',
  'cancel_relation',
  'note',
  'react_entity',
  'roll_luck',
] as const;

const LORE_HITS = 5;

/** What roll_luck tells the master: the outcome; what it is, he decides (ADR 47). */
const LUCK_TEXT: Record<LuckOutcome, (stakes: Stakes) => string> = {
  success: (stakes) =>
    `Luck: it works out. Narrate what the player finds or gains: fitting the place and the story, ${stakes} at most. ` +
    'Just tell it; the world records it after your reply.',
  failure: () => 'Luck: it does not work out. Nothing is found or gained; narrate that plainly.',
  mishap: () =>
    'Luck: bad luck. It fails, and something small goes wrong for the player, fitting the place: a noise, a cut, ' +
    'a watchful eye, something dropped. You decide what; keep it small and narrate it.',
};

export function createGmTools({ world, embed, actor, luck = DEFAULT_LUCK, random = Math.random }: GmToolDeps): StructuredToolInterface[] {
  // The agent may run several tool calls of one model reply concurrently; world commands must
  // apply in the order the model wrote them (move, then take).
  const run = serial();
  const guarded =
    <A>(name: string, body: (args: A, signal?: AbortSignal) => Promise<string>) =>
    (args: A, config?: { signal?: AbortSignal }) =>
      run(async () => {
        try {
          return await body(args, config?.signal);
        } catch (error) {
          const reason = (error as Error).message;
          return `Error: ${name} failed unexpectedly (${reason}). Narrate around it; do not retry.`;
        }
      });

  const inspect = tool(
    guarded('inspect', async ({ target }: { target: string }) => {
      const visible = await world.state.visible();
      const found = world.state.resolve(target, visible);
      if (found.kind === 'ambiguous') {
        const candidates = found.candidates.map((e) => `${e.name} [${e.id}]`).join(', ');
        return `Error: "${target}" is ambiguous: ${candidates}.`;
      }
      const entity = found.kind === 'found' ? found.entity : await world.state.entity(target);
      if (!entity) return `Error: there is no "${target}" here or in the inventory.`;
      return describe(world, entity);
    }),
    {
      name: 'inspect',
      description:
        'Full details of something the player can see or has: a person, a thing, the location, the player. ' +
        'Shows its whole description, what it holds or contains, and where it is. Changes nothing.',
      schema: z.object({ target: z.string().describe('Id or name') }),
    },
  );

  const searchLore = tool(
    guarded('search_lore', async ({ query }: { query: string }) => {
      if (!embed) return 'Error: lore search is not available right now. Rely on the world state.';
      const hits = await world.searchLore(await embed(query), LORE_HITS);
      if (!hits.ok) return `Error: ${hits.error}`;
      if (hits.value.length === 0) return 'Nothing in the lore matches.';
      return hits.value
        .map(({ entity: e }) => `- ${e.name} [${e.id}] (${e.type.toLowerCase()}): ${brief(e.description, 400)}`)
        .join('\n');
    }),
    {
      name: 'search_lore',
      description:
        'Search the story\'s lore by meaning: people, places, things and their history or secrets, ' +
        'including what is not in the current scene. Use it before inventing facts about this world.',
      schema: z.object({ query: z.string().describe('What you want to know, in a few words') }),
    },
  );

  const setEntity = tool(
    guarded('set_entity', async (args: { id?: string; name?: string; type?: Creatable; description?: string }) =>
      result(await world.setEntity(args), (e) => said(e)),
    ),
    {
      name: 'set_entity',
      description:
        'Make something real, or change it. With id: change the fields you give of anything — a thing, a person, ' +
        'a place, or the player (id "player": wounds, clothes, disguise, reputation); a description replaces the ' +
        'old one whole. Without id: create something new in the current location — name, type and description ' +
        "are all required; it persists, and set_relation can put it in someone's hands. Use it as soon as the " +
        'story makes something real: what an honest character said is there, what the player finds or makes. ' +
        'Nothing powerful or story-breaking, never places.',
      schema: z.object({
        id: z.string().optional().describe('Id or name of what to change; leave it out to create something new'),
        name: z.string().optional().describe('Short unique name, e.g. "bag of tools" (required to create)'),
        type: z
          .enum(CREATABLE)
          .optional()
          .describe('Required to create. item or document: can be carried; npc, creature; other: anything else'),
        description: z.string().optional().describe('The whole description: what it is, looks like, its state'),
      }),
    },
  );

  const setRelation = tool(
    guarded(
      'set_relation',
      async ({ a, relation, b, description }: { a: string; relation: RelationName; b: string; description?: string }) =>
        result(await world.setRelation(a, relation.toUpperCase(), b, description), (e) => said(e)),
    ),
    {
      name: 'set_relation',
      description:
        'Make a relation true; the world checks its rules and refuses what cannot happen. ' +
        'located_in — a is in b: the player goes to b (only along a listed exit); a person or creature goes to ' +
        'any location, in sight or not; something is put down here or into a container. ' +
        'has_item — a carries b: the player takes something here (from a person here only after react_entity and ' +
        'a played-out attempt); someone gets or is given a thing. ' +
        "interacts_with — how a stands with b now (allegiance, grudge, debt, trust), told from a's side; it " +
        'replaces the old tie between them. has_attribute — a lasting quality b of a. ' +
        'Something is in one place at a time: the old place goes by itself.',
      schema: z.object({
        a: z.string().describe('Id or name: who or what the relation is about ("player" for the player)'),
        relation: z.enum(RELATIONS),
        b: z.string().describe('Id or name of the other side: a place, a container, a thing, a person'),
        description: z
          .string()
          .optional()
          .describe(
            'For interacts_with and has_attribute: the tie as it is now, e.g. ' +
              '"the stranger refused to join the heist; she no longer trusts them"',
          ),
      }),
    },
  );

  const cancelRelation = tool(
    guarded('cancel_relation', async ({ a, b }: { a: string; b: string }) =>
      result(await world.cancelRelation(a, b), (e) => said(e)),
    ),
    {
      name: 'cancel_relation',
      description:
        'End a tie that is no longer true — someone left a crew, broke an alliance, no longer serves or is owed. ' +
        'Use it after set_relation, when the old tie stays behind as a lie (set_relation already rewrites a tie ' +
        'of the same kind between the two). People stop acting on it.',
      schema: z.object({
        a: z.string().describe('Id or name of one side'),
        b: z.string().describe('Id or name of the other side'),
      }),
    },
  );

  const note = tool(
    guarded('note', async ({ text, close_id }: { text?: string; close_id?: number }) => {
      if (close_id !== undefined) {
        return result(await world.closeNote(close_id), async () => `Note #${close_id} is closed.`);
      }
      if (!text?.trim()) return 'Error: give the text of a new note, or close_id to close one.';
      return result(await world.addNote(text), async (e) =>
        e.kind === 'note.added' ? `Note #${e.payload.id} is added.` : 'Done.',
      );
    }),
    {
      name: 'note',
      description:
        'Your notebook, kept in the world state even when the conversation is summarized away. ' +
        'Add a note for a quest, promise, clue, debt or plot thread as soon as it appears; ' +
        'close it by id once it is resolved.',
      schema: z.object({
        text: z.string().optional().describe('Text of a new note'),
        close_id: z.number().int().optional().describe('Id of an open note to close instead'),
      }),
    },
  );

  const reactEntityTool = tool(
    guarded('react_entity', async ({ target }: { target: string; situation: string }) => {
      if (!actor) return 'Error: nothing can react right now. Narrate around it.';
      const nearby = (await world.state.visible()).filter((e) => e.type !== 'LOCATION' && e.type !== 'PLAYER');
      const found = world.state.resolve(target, nearby);
      if (found.kind === 'ambiguous') {
        const candidates = found.candidates.map((e) => `${e.name} [${e.id}]`).join(', ');
        return `Error: "${target}" is ambiguous: ${candidates}.`;
      }
      if (found.kind === 'none') return `Error: there is no "${target}" here to react.`;
      return reactionAsked(found.entity);
    }),
    {
      name: 'react_entity',
      description:
        'Let something here react, in its own words and/or deeds, to what just happened: a person, a creature, ' +
        'anything that might perceive (a magic mirror, a watchful statue) — here or in the inventory. Use it ' +
        'whenever someone or something present would respond to the player or the scene: a question, a shove, a ' +
        'bribe, a grab at their purse — instead of writing their lines yourself. It speaks to the player in its ' +
        'own voice right after this; a plain object stays silent. Several calls react one after another.',
      schema: z.object({
        target: z.string().describe('Id or name of who or what reacts (here or in the inventory)'),
        situation: z
          .string()
          .describe(
            'What just happened, told to them in the second person, e.g. "The stranger asks you about the vault" ' +
              'or "The stranger grabs the brass key from your belt"',
          ),
      }),
    },
  );

  const rollLuckTool = tool(
    guarded('roll_luck', async (input: { attempt: string; likelihood: Likelihood; stakes: Stakes }) => {
      const roll = rollLuck(input, luck, random);
      const recorded = await world.recordLuck(roll);
      if (!recorded.ok) return `Error: ${recorded.error}`;
      return LUCK_TEXT[roll.outcome](roll.stakes);
    }),
    {
      name: 'roll_luck',
      description:
        "Roll the player's luck when the outcome of what they try is genuinely uncertain and matters: searching " +
        'for something useful (a coin under the table, a spare key), hoping for a lucky break, a small risky ' +
        'move. Not for what the place makes certain (a bartender in a bar), not for what the world rules decide ' +
        '(exits, what is here), not for what people decide (react_entity). Call it before you narrate the ' +
        'outcome; then narrate it as the result says.',
      schema: z.object({
        attempt: z.string().describe('What the player tries or hopes for, e.g. "search under the tables for anything useful"'),
        likelihood: z.enum(LIKELIHOODS).describe('How likely it is to work out here, as you judge the place and the moment'),
        stakes: z.enum(STAKES).describe('What they stand to gain: trivial (a coin), useful (a spare key), valuable (real help)'),
      }),
    },
  );

  /** What a world change means, for the model (names with ids). */
  async function said(e: WorldEvent): Promise<string> {
    const names = new NameBook((id) => world.state.entity(id));
    const name = (id: string) => names.of(id);
    const n = async (id: string) => `${await name(id)} [${id}]`;
    switch (e.kind) {
      case 'player.moved':
        return `The player is now in ${await n(e.payload.to)}. The world state shows the new scene.`;
      case 'npc.moved':
        return `${await n(e.payload.npc)} is now in ${await n(e.payload.to)}.`;
      case 'item.taken':
        return `The player now has ${await n(e.payload.item)}${
          e.payload.from ? ` (taken from ${await name(e.payload.from)})` : ''
        }.`;
      case 'item.dropped':
        return `${await n(e.payload.item)} now lies in ${await name(e.payload.to)}.`;
      case 'item.moved':
        return `${await n(e.payload.item)} is now with ${await n(e.payload.to)}.`;
      case 'entity.added':
        return `Created ${await n(e.payload.entity)} here. It is now a lasting part of the world.`;
      case 'entity.edited':
        return `${await n(e.payload.id)} is updated.`;
      case 'relation.set':
        return `${await name(e.payload.src)} — ${await name(e.payload.dst)}: ${
          e.payload.before === null ? 'tie recorded' : 'tie rewritten'
        }. They will act on it.`;
      case 'relation.cancelled':
        return `${await name(e.payload.a)} and ${await name(e.payload.b)} are no longer tied.`;
      default:
        return 'Done.';
    }
  }

  return [inspect, searchLore, setEntity, setRelation, cancelRelation, note, reactEntityTool, rollLuckTool];
}

const CREATABLE = lowerTypes(ADDABLE_TYPES);
type Creatable = (typeof CREATABLE)[number];
const RELATIONS = ['located_in', 'has_item', 'interacts_with', 'has_attribute'] as const;
type RelationName = (typeof RELATIONS)[number];

async function result(outcome: Outcome, success: (event: WorldEvent) => Promise<string>): Promise<string> {
  return outcome.ok ? success(outcome.value) : `Error: ${outcome.error}`;
}

/** What `inspect` tells about one entity. */
async function describe(world: World, entity: Entity): Promise<string> {
  const { state } = world;
  const list = (entities: Entity[]) => entities.map((e) => `${e.name} [${e.id}]`).join(', ');
  const fixed = entity.props.fixed === true ? ', fixed in place' : '';
  const lines = [
    `${entity.name} [${entity.id}] (${entity.type.toLowerCase()}${fixed})`,
    entity.description || 'No description.',
  ];
  if (entity.type === 'LOCATION') {
    const exits = (await state.exits(entity.id)).map(
      (x) => `${x.to.name} [${x.to.id}]${x.description ? ` — ${x.description}` : ''}`,
    );
    lines.push(`Exits: ${exits.join('; ') || 'none'}.`);
    const contents = await state.contents(entity.id);
    if (contents.length) lines.push(`Here: ${list(contents)}.`);
    return lines.join('\n');
  }
  const places = await Promise.all((await state.placesOf(entity.id)).map((id) => state.entity(id)));
  const where = places.filter((e): e is Entity => !!e);
  const holder = where.find(isCarrier);
  if (holder) lines.push(`Carried by: ${holder.name} [${holder.id}].`);
  else if (where.length) lines.push(`Where: ${list(where)}.`);
  const being = isCarrier(entity);
  const held = being ? await state.inventory(entity.id) : await state.contents(entity.id);
  if (held.length) lines.push(`${being ? 'Carries' : 'Contains'}: ${list(held)}.`);
  return lines.join('\n');
}

/** Runs async jobs one at a time, in call order; a failed job does not block the next. */
function serial(): <T>(job: () => Promise<T>) => Promise<T> {
  let tail: Promise<unknown> = Promise.resolve();
  return (job) => {
    const next = tail.then(job);
    tail = next.catch(() => undefined);
    return next;
  };
}
