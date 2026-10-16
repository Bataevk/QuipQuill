import type { Db } from '../storage/db';
import type { VectorIndex } from '../vector/vector-index';
import { putEntity, relate, unlocate, unplace } from './dynamic-graph';
import { EventLog } from './event-log';
import {
  fail,
  ok,
  type Actor,
  type Entity,
  type EntityPatch,
  type EntityType,
  type Exit,
  type Note,
  type Outcome,
  type Props,
  type WorldEvent,
} from './model';
import { normalizeName, slugify } from './names';
import { WorldState, type Resolution } from './state';

export interface CharacterSheet {
  id: string | null;
  name: string;
  sex?: string | null;
  age?: number | null;
  race?: string | null;
  description: string;
}

/** What the master sees each call ({{WORLD_STATE}}, agent.md). */
export interface Scene {
  player: Entity;
  location: Entity | undefined;
  exits: Exit[];
  here: Entity[];
  inventory: Entity[];
  notes: Note[];
  recent: WorldEvent[];
}

export interface LoreHit {
  entity: Entity;
  score: number;
}

/** Entity types the master may add to a scene (the map itself comes only from the story). */
const ADDABLE: readonly EntityType[] = ['ITEM', 'DOCUMENT', 'NPC', 'CREATURE', 'OTHER'];
const ID_PREFIX: Record<string, string> = {
  ITEM: 'item',
  DOCUMENT: 'doc',
  NPC: 'npc',
  CREATURE: 'npc',
  OTHER: 'ent',
};
const PORTABLE: readonly EntityType[] = ['ITEM', 'DOCUMENT'];
export const RECENT_EVENTS = 5;

/** A broken rule inside a transaction: rolls it back and becomes a failed Outcome. */
class Rejected extends Error {
  readonly reason: string;

  constructor(reason: string) {
    super(reason);
    this.reason = reason;
  }
}

type Tx = { db: Db; state: WorldState; log: EventLog };

/**
 * The playthrough's world over an open save. Every change is a command: rules are checked, then
 * the event and the change to the dynamic graph `dyn_*` are written in one transaction
 * (data-model.md#события); the story's own tables are never written (ADR 42).
 * Broken rules come back as `{ ok: false, error }` with text meant for the model.
 */
export class World {
  readonly state: WorldState;
  readonly log: EventLog;
  private readonly db: Db;
  private readonly vectors: VectorIndex | undefined;
  private turn = 0;

  private constructor(db: Db, vectors: VectorIndex | undefined) {
    this.db = db;
    this.vectors = vectors;
    this.state = new WorldState(db);
    this.log = new EventLog(db);
  }

  static async open(db: Db, options: { vectors?: VectorIndex } = {}): Promise<World> {
    const world = new World(db, options.vectors);
    world.turn = await world.log.lastTurn();
    return world;
  }

  get currentTurn(): number {
    return this.turn;
  }

  /** Starts the next player turn; events written until the next call carry this number. */
  beginTurn(): number {
    this.turn += 1;
    return this.turn;
  }

  /** Goes back to an unfinished turn (a retry): its events keep one number. */
  continueTurn(turn: number): number {
    this.turn = turn;
    return turn;
  }

  // --- queries -------------------------------------------------------------------------------

  async scene(): Promise<Scene> {
    const player = await this.state.player();
    if (!player) throw new Error('The session has not started');
    const location = await this.state.locationOf(player.id);
    return {
      player,
      location,
      exits: location ? await this.state.exits(location.id) : [],
      here: location ? await this.state.contents(location.id) : [],
      inventory: await this.state.inventory(player.id),
      notes: await this.log.openNotes(),
      recent: await this.log.recent(RECENT_EVENTS),
    };
  }

  /** Lore search by meaning (story embeddings), best match per entity. */
  async searchLore(query: Float32Array, limit = 5): Promise<Outcome<LoreHit[]>> {
    if (!this.vectors) return fail('Lore search is not available in this playthrough.');
    const hits = await this.vectors.search(query, limit * 2);
    const best = new Map<string, number>();
    for (const hit of hits) best.set(hit.entityId, Math.max(best.get(hit.entityId) ?? -1, hit.score));
    const result: LoreHit[] = [];
    for (const [id, score] of best) {
      const entity = await this.state.entity(id);
      if (entity) result.push({ entity, score });
    }
    return ok(result.sort((a, b) => b.score - a.score).slice(0, limit));
  }

  // --- commands ------------------------------------------------------------------------------

  /** `session.started`: the player enters the dynamic graph at the start location. */
  startSession(input: {
    storyId: string;
    character: CharacterSheet;
    startLocation: string;
  }): Promise<Outcome> {
    return this.command('system', async ({ db, state, log }) => {
      if ((await log.count('session.started')) > 0) reject('The session has already started.');
      const playerId = `player:${input.character.id ?? 'guest'}`;
      const event = await log.append(this.turn, 'system', 'session.started', {
        story_id: input.storyId,
        character_id: input.character.id,
        player: playerId,
      });
      const start = await state.entity(input.startLocation);
      if (start?.type !== 'LOCATION') reject(`Start location ${input.startLocation} does not exist.`);
      const { id: _id, name, description, ...sheet } = input.character;
      await putEntity(db, event.seq, {
        id: playerId,
        type: 'PLAYER',
        subtype: null,
        name,
        description,
        props: { ...sheet, original: input.character },
      });
      await relate(db, event.seq, playerId, input.startLocation, 'LOCATED_IN');
      return event;
    });
  }

  /** `character.adapted`: the player's choice after `propose_character` (stage 4.4), once. */
  adaptCharacter(input: {
    adapted: { name?: string; race?: string; age?: number | null; description?: string };
    choice: string;
  }): Promise<Outcome> {
    return this.command('gm', async ({ db, log }, player) => {
      if ((await log.count('character.adapted')) > 0) reject('The character is already settled.');
      const original = (player.props.original as Props | undefined) ?? {};
      const event = await log.append(this.turn, 'gm', 'character.adapted', {
        original,
        adapted: input.adapted,
        choice: input.choice,
      });
      const { race, age } = input.adapted;
      const props = { ...player.props, ...(race ? { race } : {}), ...(typeof age === 'number' ? { age } : {}) };
      await putEntity(db, event.seq, {
        ...player,
        name: input.adapted.name ?? player.name,
        description: input.adapted.description ?? player.description,
        props,
      });
      return event;
    });
  }

  /** `player.moved`: only along a `LEADS_TO` path from the current location. */
  move(target: string): Promise<Outcome> {
    return this.command('gm', async ({ db, state, log }, player) => {
      const here = await state.locationOf(player.id);
      if (!here) reject('The player is nowhere; the session has not placed them.');
      const exits = await state.exits(here.id);
      const to = pick(
        state.resolve(target, exits.map((exit) => exit.to)),
        target,
        () =>
          `There is no path from ${here.name} to "${target}". Exits: ${
            exits.map((exit) => describeExit(exit)).join('; ') || 'none'
          }.`,
      );
      const event = await log.append(this.turn, 'gm', 'player.moved', { from: here.id, to: to.id });
      await unlocate(db, event.seq, player.id);
      await relate(db, event.seq, player.id, to.id, 'LOCATED_IN');
      return event;
    });
  }

  /**
   * `item.taken`: a portable item that lies here, or inside something that lies here. An item
   * an NPC here holds only after a played-out attempt (ADR 32, 34): this turn already has that
   * NPC's `npc.reacted`, and the master decided the attempt worked.
   */
  take(target: string): Promise<Outcome> {
    return this.command('gm', async ({ db, state, log }, player) => {
      const owned = state.resolve(target, await state.inventory(player.id));
      if (owned.kind === 'found') reject(`The player already has ${owned.entity.name}.`);
      const reachable = await this.reachable(state, player);
      const held = await this.heldByNpcsHere(state, player);
      const item = pick(
        state.resolve(target, [...reachable, ...held.map((h) => h.item)]),
        target,
        () => `There is no "${target}" here to take.`,
      );
      if (!PORTABLE.includes(item.type)) reject(`${item.name} cannot be carried.`);
      if (item.props.fixed === true) reject(`${item.name} is fixed in place and cannot be taken.`);
      const holder = held.find((h) => h.item.id === item.id)?.npc;
      if (holder && !(await log.reacted(this.turn, holder.id))) {
        reject(
          `${holder.name} has ${item.name}. Play out the attempt first: call npc_react for ` +
            `${holder.name}, then take it only if the attempt succeeds.`,
        );
      }
      const [from = ''] = await state.placesOf(item.id);
      const event = await log.append(this.turn, 'gm', 'item.taken', { item: item.id, from });
      await unplace(db, event.seq, item.id);
      await relate(db, event.seq, player.id, item.id, 'HAS_ITEM');
      return event;
    });
  }

  /** `item.dropped`: from the inventory to the current location. */
  drop(target: string): Promise<Outcome> {
    return this.command('gm', async ({ db, state, log }, player) => {
      const inventory = await state.inventory(player.id);
      const item = pick(state.resolve(target, inventory), target, () =>
        `The player does not have "${target}". Inventory: ${names(inventory) || 'empty'}.`,
      );
      const here = await state.locationOf(player.id);
      if (!here) reject('The player is nowhere; the session has not placed them.');
      const event = await log.append(this.turn, 'gm', 'item.dropped', { item: item.id, to: here.id });
      await unplace(db, event.seq, item.id);
      await relate(db, event.seq, item.id, here.id, 'LOCATED_IN');
      return event;
    });
  }

  /** `entity.added`: something new in the current scene (or in the player's hands). */
  addEntity(input: {
    name: string;
    type: EntityType;
    description: string;
    inInventory?: boolean;
  }): Promise<Outcome> {
    return this.command('gm', async ({ db, state, log }, player) => {
      const name = input.name.trim();
      if (!name) reject('The new entity needs a name.');
      if (!ADDABLE.includes(input.type)) {
        reject(`Cannot add a ${input.type}; allowed types: ${ADDABLE.join(', ')}.`);
      }
      if (input.inInventory && !PORTABLE.includes(input.type)) {
        reject(`A ${input.type} cannot be put in the inventory.`);
      }
      const all = await state.allEntities();
      const clash = all.find((e) => normalizeName(e.name) === normalizeName(name));
      if (clash) reject(`Something named "${clash.name}" already exists (${clash.id}).`);
      const here = await state.locationOf(player.id);
      if (!here) reject('The player is nowhere; the session has not placed them.');

      const ids = new Set(all.map((e) => e.id));
      const base = `${ID_PREFIX[input.type]}:${slugify(name)}`;
      let id = base;
      for (let n = 2; ids.has(id); n++) id = `${base}-${n}`;
      const relation = input.inInventory
        ? { type: 'HAS_ITEM', target: player.id }
        : { type: 'LOCATED_IN', target: here.id };
      const event = await log.append(this.turn, 'gm', 'entity.added', { entity: id, relation });
      await putEntity(db, event.seq, {
        id,
        type: input.type,
        subtype: null,
        name,
        description: input.description,
        props: { added: true },
      });
      if (input.inInventory) await relate(db, event.seq, player.id, id, 'HAS_ITEM');
      else await relate(db, event.seq, id, here.id, 'LOCATED_IN');
      return event;
    });
  }

  /** `entity.edited`: description and/or props (merged) of a visible entity, or any by id. */
  editEntity(target: string, patch: EntityPatch): Promise<Outcome> {
    return this.command('gm', async ({ db, state, log }) => {
      if (patch.description === undefined && patch.props === undefined) reject('Nothing to change.');
      const entity =
        (await state.entity(target)) ??
        pick(state.resolve(target, await state.visible()), target, () => `"${target}" is not here.`);
      if (entity.type === 'PLAYER') reject('Change the player with update_player.');
      const before: EntityPatch = {};
      const after: EntityPatch = {};
      if (patch.description !== undefined) {
        before.description = entity.description;
        after.description = patch.description;
      }
      if (patch.props !== undefined) {
        before.props = entity.props;
        after.props = { ...entity.props, ...patch.props };
      }
      const event = await log.append(this.turn, 'gm', 'entity.edited', { id: entity.id, before, after });
      await putEntity(db, event.seq, {
        ...entity,
        description: after.description ?? entity.description,
        props: after.props ?? entity.props,
      });
      return event;
    });
  }

  /** `player.updated`: the player's description (wounds, clothes, reputation…). */
  updatePlayer(description: string): Promise<Outcome> {
    return this.command('gm', async ({ db, log }, player) => {
      const event = await log.append(this.turn, 'gm', 'player.updated', {
        before: player.description,
        after: description,
      });
      await putEntity(db, event.seq, { ...player, description });
      return event;
    });
  }

  /** `note.added`: a quest, promise or plot thread the master must not forget. */
  addNote(text: string): Promise<Outcome> {
    return this.command('gm', async ({ db, log }) => {
      if (!text.trim()) reject('The note is empty.');
      const row = await db.get<{ id: number }>('SELECT coalesce(max(id), 0) + 1 AS id FROM notes');
      const id = Number(row?.id ?? 1);
      const event = await log.append(this.turn, 'gm', 'note.added', { id, text: text.trim() });
      await db.run('INSERT INTO notes (id, text, created_seq) VALUES (?, ?, ?)', [id, text.trim(), event.seq]);
      return event;
    });
  }

  /** `note.closed`: the thread is resolved. */
  closeNote(id: number): Promise<Outcome> {
    return this.command('gm', async ({ db, log }) => {
      const note = await log.note(id);
      if (!note) reject(`There is no note #${id}.`);
      if (note.closed) reject(`Note #${id} is already closed.`);
      const event = await log.append(this.turn, 'gm', 'note.closed', { id, text: note.text });
      await db.run('UPDATE notes SET closed_seq = ? WHERE id = ?', [event.seq, id]);
      return event;
    });
  }

  /**
   * `npc.reacted`: journal only — the NPC's own chat keeps the conversation (stage 4.3). The
   * situation is what happened from the NPC's side (a line, a grab, a noise); the reaction is
   * its words and/or deeds, or "[silence]".
   */
  recordNpcReaction(input: { npc: string; situation: string; reaction: string }): Promise<Outcome> {
    return this.command('gm', async ({ state, log }) => {
      if (!(await state.entity(input.npc))) reject(`There is no ${input.npc}.`);
      return log.append(this.turn, 'gm', 'npc.reacted', input);
    });
  }

  // --- internals -----------------------------------------------------------------------------

  /** Items the player can reach: here, and one level inside containers that are here. */
  private async reachable(state: WorldState, player: Entity): Promise<Entity[]> {
    const here = await state.locationOf(player.id);
    if (!here) return [];
    const direct = await state.contents(here.id);
    const nested = await Promise.all(
      direct.filter((e) => PORTABLE.includes(e.type) || e.type === 'OTHER').map((e) => state.contents(e.id)),
    );
    return [...direct, ...nested.flat()];
  }

  /** Items that NPCs and creatures in the player's location carry. */
  private async heldByNpcsHere(
    state: WorldState,
    player: Entity,
  ): Promise<{ npc: Entity; item: Entity }[]> {
    const here = await state.locationOf(player.id);
    if (!here) return [];
    const npcs = (await state.contents(here.id)).filter((e) => e.type === 'NPC' || e.type === 'CREATURE');
    const held = await Promise.all(
      npcs.map(async (npc) => (await state.inventory(npc.id)).map((item) => ({ npc, item }))),
    );
    return held.flat();
  }

  private async command(
    actor: Actor,
    body: (tx: Tx, player: Entity) => Promise<WorldEvent>,
  ): Promise<Outcome> {
    try {
      return ok(
        await this.db.transaction(async (db) => {
          const tx = { db, state: new WorldState(db), log: new EventLog(db) };
          const player = actor === 'system' ? undefined : await tx.state.player();
          if (actor !== 'system' && !player) reject('The session has not started.');
          return body(tx, player as Entity);
        }),
      );
    } catch (error) {
      if (error instanceof Rejected) return fail(error.reason);
      throw error;
    }
  }
}

function reject(reason: string): never {
  throw new Rejected(reason);
}

function pick(resolution: Resolution, target: string, notFound: () => string): Entity {
  if (resolution.kind === 'found') return resolution.entity;
  if (resolution.kind === 'ambiguous') {
    reject(
      `"${target}" is ambiguous: ${resolution.candidates.map((e) => `${e.name} (${e.id})`).join(', ')}.`,
    );
  }
  return reject(notFound());
}

function names(entities: Entity[]): string {
  return entities.map((e) => e.name).join(', ');
}

function describeExit(exit: Exit): string {
  return exit.description ? `${exit.to.name} (${exit.description})` : exit.to.name;
}
