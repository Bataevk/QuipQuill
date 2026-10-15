/**
 * The game master's tools (agent.md#инструменты-мастера): thin wrappers over `World`
 * commands (ADR 29). A broken rule or any failure comes back to the model as text starting with
 * "Error:", never as an exception. Tool descriptions are part of the contract with the model,
 * as much as the prompt is.
 */
import type { StructuredToolInterface } from '@langchain/core/tools';
import { tool } from 'langchain';
import { z } from 'zod';

import type { Entity, EntityType, Outcome, WorldEvent } from '../world/model';
import type { World } from '../world/world';
import { type NpcDeps, reactNpc } from './npc';
import { brief } from './prompt';

export interface GmToolDeps {
  world: World;
  /** Query vector for search_lore, with the story's `meta.embed_model` (CloudEmbeddings). */
  embed?: (text: string) => Promise<Float32Array>;
  /** NPC models and stores for npc_react; without them NPCs cannot react. */
  npc?: NpcDeps;
}

export const GM_TOOL_NAMES = [
  'move',
  'inspect',
  'take',
  'drop',
  'search_lore',
  'add_entity',
  'edit_entity',
  'update_player',
  'note',
  'npc_react',
] as const;

const LORE_HITS = 5;

export function createGmTools({ world, embed, npc }: GmToolDeps): StructuredToolInterface[] {
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
  const name = async (id: string) => (await world.state.entity(id))?.name ?? id;

  const move = tool(
    guarded('move', async ({ to }: { to: string }) =>
      result(await world.move(to), async (e) =>
        e.kind === 'player.moved'
          ? `The player is now in ${await name(e.payload.to)} [${e.payload.to}]. ` +
            'The world state shows the new scene.'
          : 'Done.',
      ),
    ),
    {
      name: 'move',
      description:
        'Move the player to an adjacent location, along one of the exits listed in the world state. ' +
        'Call it only when the player goes there; then describe the arrival from the updated world state.',
      schema: z.object({ to: z.string().describe('Exit id (loc:…) or name of the destination') }),
    },
  );

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

  const take = tool(
    guarded('take', async ({ item }: { item: string }) =>
      result(await world.take(item), async (e) =>
        e.kind === 'item.taken'
          ? `The player now has ${await name(e.payload.item)} [${e.payload.item}]` +
            (e.payload.from ? ` (taken from ${await name(e.payload.from)}).` : '.')
          : 'Done.',
      ),
    ),
    {
      name: 'take',
      description:
        'Put an item into the player\'s inventory: one lying here, or inside a container that is here. ' +
        'Things fixed in place, people and locations cannot be taken. Something an NPC here carries ' +
        'can be taken only after a played-out attempt: first npc_react for that NPC, then take only if ' +
        'you decide the attempt succeeded.',
      schema: z.object({ item: z.string().describe('Item id or name') }),
    },
  );

  const drop = tool(
    guarded('drop', async ({ item }: { item: string }) =>
      result(await world.drop(item), async (e) =>
        e.kind === 'item.dropped'
          ? `${await name(e.payload.item)} [${e.payload.item}] now lies in ${await name(e.payload.to)}.`
          : 'Done.',
      ),
    ),
    {
      name: 'drop',
      description: 'Leave an item from the inventory in the current location (dropped, put down, set aside).',
      schema: z.object({ item: z.string().describe('Item id or name from the inventory') }),
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

  const addEntity = tool(
    guarded(
      'add_entity',
      async (args: { name: string; type: Addable; description: string; place?: 'here' | 'inventory' }) =>
        result(
          await world.addEntity({
            name: args.name,
            type: args.type.toUpperCase() as EntityType,
            description: args.description,
            inInventory: args.place === 'inventory',
          }),
          async (e) =>
            e.kind === 'entity.added'
              ? `Added ${await name(e.payload.entity)} [${e.payload.entity}] ${
                  e.payload.relation.type === 'HAS_ITEM' ? 'to the inventory' : 'here'
                }. It is now a lasting part of the world.`
              : 'Done.',
        ),
    ),
    {
      name: 'add_entity',
      description:
        'Make a new detail real in the current scene, or in the player\'s hands: a small plausible thing ' +
        'the player finds or makes (a loose brick, a makeshift torch), a passer-by, a stray cat. ' +
        'It persists afterwards. Not for places, and not for anything powerful or story-breaking.',
      schema: z.object({
        name: z.string().describe('Short unique name, e.g. "makeshift torch"'),
        type: z
          .enum(ADDABLE)
          .describe('item or document: can be carried; npc, creature; other: anything else'),
        description: z.string().describe('What it is and looks like, one or two sentences'),
        place: z.enum(['here', 'inventory']).optional().describe('Default: here'),
      }),
    },
  );

  const editEntity = tool(
    guarded('edit_entity', async ({ target, description }: { target: string; description: string }) =>
      result(await world.editEntity(target, { description }), async (e) =>
        e.kind === 'entity.edited' ? `${await name(e.payload.id)} [${e.payload.id}] is updated.` : 'Done.',
      ),
    ),
    {
      name: 'edit_entity',
      description:
        'Record a lasting change to a thing, person or place: broken, opened, lit, angry, gone quiet. ' +
        'Give the whole new description (it replaces the old one). Not for the player: use update_player.',
      schema: z.object({
        target: z.string().describe('Id or name'),
        description: z.string().describe('The full new description'),
      }),
    },
  );

  const updatePlayer = tool(
    guarded('update_player', async ({ description }: { description: string }) =>
      result(await world.updatePlayer(description), async () => "The player's description is updated."),
    ),
    {
      name: 'update_player',
      description:
        "Record a lasting change to the player character: wounds, clothes, disguise, reputation. " +
        'Give the whole new description (it replaces the old one).',
      schema: z.object({ description: z.string().describe("The player character's full new description") }),
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

  const npcReact = tool(
    guarded('npc_react', async ({ target, situation }: { target: string; situation: string }, signal) => {
      if (!npc) return 'Error: people cannot react right now. Narrate around it.';
      const nearby = (await world.state.visible()).filter((e) => e.type !== 'LOCATION' && e.type !== 'PLAYER');
      const found = world.state.resolve(target, nearby);
      if (found.kind === 'ambiguous') {
        const candidates = found.candidates.map((e) => `${e.name} [${e.id}]`).join(', ');
        return `Error: "${target}" is ambiguous: ${candidates}.`;
      }
      if (found.kind === 'none') return `Error: there is no "${target}" here to react.`;
      try {
        const reaction = await reactNpc(world, npc, found.entity, situation, signal);
        const who = `${found.entity.name} [${found.entity.id}]`;
        return reaction.silent
          ? `${who} does not react.`
          : `${who} reacted; the player already saw it, so do not repeat or paraphrase it — ` +
              `narrate only its consequences and the rest of the scene:\n${reaction.text}`;
      } catch (error) {
        if (signal?.aborted) throw error;
        return `Error: ${found.entity.name} could not react right now (${(error as Error).message}). Narrate around it.`;
      }
    }),
    {
      name: 'npc_react',
      description:
        'Let a person or creature here react — in their own words and/or deeds — to what just happened: ' +
        'something the player said or asked them, did to them (a shove, a bribe, a grab at their purse), ' +
        'or an event in the scene. Use it whenever someone present would respond, instead of writing ' +
        'their lines yourself. The player sees the reaction at once. Several calls run one after another.',
      schema: z.object({
        target: z.string().describe('Id or name of who reacts (someone here)'),
        situation: z
          .string()
          .describe(
            'What just happened, told to them in the second person, e.g. "The stranger asks you about the vault" ' +
              'or "The stranger grabs the brass key from your belt"',
          ),
      }),
    },
  );

  return [move, inspect, take, drop, searchLore, addEntity, editEntity, updatePlayer, note, npcReact];
}

const ADDABLE = ['item', 'document', 'npc', 'creature', 'other'] as const;
type Addable = (typeof ADDABLE)[number];

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
  const holder = where.find((e) => BEINGS.includes(e.type));
  if (holder) lines.push(`Carried by: ${holder.name} [${holder.id}].`);
  else if (where.length) lines.push(`Where: ${list(where)}.`);
  const being = BEINGS.includes(entity.type);
  const held = being ? await state.inventory(entity.id) : await state.contents(entity.id);
  if (held.length) lines.push(`${being ? 'Carries' : 'Contains'}: ${list(held)}.`);
  return lines.join('\n');
}

/** Those who carry things (`HAS_ITEM` = inventory) rather than contain them. */
const BEINGS: readonly EntityType[] = ['NPC', 'CREATURE', 'PLAYER'];

/** Runs async jobs one at a time, in call order; a failed job does not block the next. */
function serial(): <T>(job: () => Promise<T>) => Promise<T> {
  let tail: Promise<unknown> = Promise.resolve();
  return (job) => {
    const next = tail.then(job);
    tail = next.catch(() => undefined);
    return next;
  };
}
