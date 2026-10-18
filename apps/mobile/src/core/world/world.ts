import type { Db } from '../storage/db';
import { nowIso } from '../util/ids';
import type { VectorIndex } from '../vector/vector-index';
import { forget, handTo, moveTo, putEntity, putIn, relate, unrelate } from './dynamic-graph';
import { EventLog } from './event-log';
import { LoreVectors, type Embed } from './lore-vectors';
import {
  ADDABLE_TYPES,
  type AddableType,
  fail,
  isAddableType,
  isBeing,
  isCarrier,
  isContainerLike,
  isPortable,
  isTieType,
  ok,
  type Actor,
  type Entity,
  type EntityPatch,
  type EventPayloads,
  type Exit,
  type Note,
  type Outcome,
  type Props,
  PLAYER_ID,
  type Tie,
  TIE_TYPES,
  type TieType,
  type WorldEvent,
} from './model';
import { normalizeName, type Resolution, resolveName, slugify } from './names';
import { WorldState } from './state';

export interface CharacterSheet {
  id: string | null;
  name: string;
  sex?: string | null;
  age?: number | null;
  race?: string | null;
  description: string;
}

/** What the master sees each call (the world state, agent.md). */
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

const ID_PREFIX: Record<AddableType, string> = {
  ITEM: 'item',
  DOCUMENT: 'doc',
  NPC: 'npc',
  CREATURE: 'npc',
  OTHER: 'ent',
};
export const RECENT_EVENTS = 5;
/** New entities the master may create in a player's turn (ADR 54); the generator adds the rest. */
export const MAX_GM_ADDITIONS = 1;

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
  private readonly lore: LoreVectors;
  private turn = 0;
  /** A player's turn: the master's new entities are limited (MAX_GM_ADDITIONS). */
  private playerTurn = false;
  /** Who the master's commands are journaled for: 'gm', or 'generator' (actingAs). */
  private actor: Actor = 'gm';

  private constructor(db: Db, lore: LoreVectors) {
    this.db = db;
    this.lore = lore;
    this.state = new WorldState(db);
    this.log = new EventLog(db);
  }

  /**
   * `vectors`: the story's index; without it there is no lore search. `embed`: computes the
   * vectors of new and rewritten entities (stage 6.2b); without it they are not searchable.
   */
  static async open(db: Db, options: { vectors?: VectorIndex; embed?: Embed } = {}): Promise<World> {
    const world = new World(db, new LoreVectors(db, options.vectors, options.embed));
    world.turn = await world.log.lastTurn();
    return world;
  }

  get currentTurn(): number {
    return this.turn;
  }

  /**
   * Starts the next turn; events written until the next call carry this number. A player's turn
   * (`player`) limits what the master creates; the opening and other system turns do not.
   */
  beginTurn(options: { player?: boolean } = {}): number {
    this.turn += 1;
    this.playerTurn = options.player ?? false;
    return this.turn;
  }

  /**
   * Runs `fn` with the master's commands journaled for another actor — the background
   * generator (stage 6.2d) uses the same rules, under the same turn number.
   */
  async actingAs<T>(actor: Actor, fn: () => Promise<T>): Promise<T> {
    const previous = this.actor;
    this.actor = actor;
    try {
      return await fn();
    } finally {
      this.actor = previous;
    }
  }

  /** Goes back to an unfinished turn (a retry): its events keep one number. */
  continueTurn(turn: number, options: { player?: boolean } = {}): number {
    this.turn = turn;
    this.playerTurn = options.player ?? false;
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

  /**
   * Lore search by meaning, best match per entity: the story's vectors and the playthrough's
   * own; an entity found by text the playthrough has rewritten is not returned (ADR 42).
   */
  async searchLore(query: Float32Array, limit = 5): Promise<Outcome<LoreHit[]>> {
    if (!this.lore.available) return fail('Lore search is not available in this playthrough.');
    const hits = await this.lore.search(query, limit * 2);
    const best = new Map<string, number>();
    for (const hit of hits) best.set(hit.entityId, Math.max(best.get(hit.entityId) ?? -1, hit.score));
    const result: LoreHit[] = [];
    for (const [id, score] of best) {
      const entity = await this.state.entity(id);
      if (entity) result.push({ entity, score });
    }
    return ok(result.sort((a, b) => b.score - a.score).slice(0, limit));
  }

  /**
   * Computes the vectors of new and rewritten entities that are still missing (after a change,
   * and when a save opens). Rejects without network; the next call tries again.
   */
  refreshLore(): Promise<number> {
    return this.lore.refresh();
  }

  // --- commands ------------------------------------------------------------------------------

  /** `session.started`: the player enters the dynamic graph at the start location. */
  startSession(input: {
    storyId: string;
    character: CharacterSheet;
    startLocation: string;
  }): Promise<Outcome> {
    return this.transact(async ({ db, state, log }) => {
      if ((await log.count('session.started')) > 0) reject('The session has already started.');
      const playerId = PLAYER_ID;
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
    return this.command(async ({ db, log }, player) => {
      if ((await log.count('character.adapted')) > 0) reject('The character is already settled.');
      const original = (player.props.original as Props | undefined) ?? {};
      const event = await log.append(this.turn, this.actor, 'character.adapted', {
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

  /**
   * `set_relation` (ADR 43): makes `a -type-> b` true, by the world's rules for that type.
   * - `LOCATED_IN` — `a` is in `b`: the player walks only along an exit (`player.moved`); a
   *   person or creature goes to any location (`npc.moved`); a thing from the player's hands is
   *   put down here or into something here (`item.dropped`); any other thing the master moves
   *   (`relation.set`). One place at a time: the old one goes.
   * - `HAS_ITEM` — `a` carries `b`: the player takes a portable, unfixed thing that is here, or
   *   one a person here holds once that person reacted this turn (`item.taken`, ADR 32, 34);
   *   someone else gets it from wherever it is — from the player only when they are here
   *   (`relation.set`). Into a place or a container it is `LOCATED_IN`.
   * - `INTERACTS_WITH`, `HAS_ATTRIBUTE` — a tie (`relation.set`, 6.2c).
   * Paths (`LEADS_TO`) come only from the story.
   */
  setRelation(a: string, type: string, b: string, description = ''): Promise<Outcome> {
    const kind = type.trim().toUpperCase();
    if (kind === 'LOCATED_IN') return this.command((tx, player) => this.place(tx, player, a, b));
    if (kind === 'HAS_ITEM') return this.command((tx, player) => this.give(tx, player, a, b));
    if (isTieType(kind)) {
      return this.command((tx, player) => this.tie(tx, player, a, b, kind, description));
    }
    return Promise.resolve(
      fail(
        kind === 'LEADS_TO'
          ? 'Paths between places come from the story; they cannot be added or moved.'
          : `Unknown relation ${type}; use LOCATED_IN, HAS_ITEM, ${TIE_TYPES.join(', ')}.`,
      ),
    );
  }

  /**
   * `set_entity` (ADR 43): with `id`, changes the fields given (name, description; the type
   * stays) of any entity, the player (`player`) too — `entity.edited`. Without `id`, creates
   * something new in the current location — name, type and description required; only small
   * things and beings, never places — `entity.added`.
   */
  setEntity(input: {
    id?: string;
    name?: string;
    type?: string;
    description?: string;
    /** A new entity as a draft (the Guard's expected gap, ADR 48): kept only if the turn uses it. */
    draft?: boolean;
  }): Promise<Outcome> {
    return this.command(async ({ db, state, log }, player) => {
      const name = input.name?.trim();
      const type = input.type?.trim().toUpperCase();
      const all = await state.allEntities();
      const clash = (exceptId?: string) =>
        name ? all.find((e) => e.id !== exceptId && normalizeName(e.name) === normalizeName(name)) : undefined;

      if (input.id?.trim()) {
        const entity = await this.find(state, input.id, () =>
          `There is no "${input.id}". To create something new, leave out id.`,
        );
        if (type && type !== entity.type) reject(`${entity.name} is a ${entity.type}; its type cannot change.`);
        const after: EntityPatch = {};
        const before: EntityPatch = {};
        if (name && name !== entity.name) {
          const other = clash(entity.id);
          if (other) reject(`Something named "${other.name}" already exists (${other.id}).`);
          before.name = entity.name;
          after.name = name;
        }
        if (input.description !== undefined && input.description !== entity.description) {
          before.description = entity.description;
          after.description = input.description;
        }
        if (Object.keys(after).length === 0) reject('Nothing to change.');
        const event = await log.append(this.turn, this.actor, 'entity.edited', { id: entity.id, before, after });
        await putEntity(db, event.seq, {
          ...entity,
          name: after.name ?? entity.name,
          description: after.description ?? entity.description,
        });
        return event;
      }

      if (!name || !type || input.description === undefined) {
        reject('To create something, give its name, type and description (or an id to change something).');
      }
      if (!isAddableType(type)) reject(`Cannot create a ${type}; allowed types: ${ADDABLE_TYPES.join(', ')}.`);
      const other = clash();
      if (other) reject(`Something named "${other.name}" already exists (${other.id}).`);
      if (this.actor === 'gm' && this.playerTurn) {
        const added = (await log.ofTurn(this.turn)).filter((e) => e.kind === 'entity.added' && e.actor === 'gm');
        if (added.length >= MAX_GM_ADDITIONS) {
          reject(
            'You already made something new real this turn. Narrate the rest as part of the scene: ' +
              'the world adds what it implies after your reply.',
          );
        }
      }
      const here = await hereOf(state, player);
      const ids = new Set(all.map((e) => e.id));
      const base = `${ID_PREFIX[type]}:${slugify(name)}`;
      let id = base;
      for (let n = 2; ids.has(id); n++) id = `${base}-${n}`;
      const event = await log.append(this.turn, this.actor, 'entity.added', {
        entity: id,
        relation: { type: 'LOCATED_IN', target: here.id },
      });
      const description = input.description;
      const props = input.draft ? { added: true, draft: true } : { added: true };
      await putEntity(db, event.seq, { id, type, subtype: null, name, description, props });
      await relate(db, event.seq, id, here.id, 'LOCATED_IN');
      return event;
    });
  }

  /**
   * `relation.cancelled`: the ties between two entities (both directions; of one type if given)
   * are no longer true — someone left a crew, broke an alliance. A story tie is cancelled, not
   * deleted (ADR 42).
   */
  cancelRelation(a: string, b: string, type?: string): Promise<Outcome> {
    return this.command(async ({ state, log, db }, player) => {
      const [one, other] = await this.pair(state, a, b);
      const ties = (await this.tiesBetween(db, one.id, other.id)).filter((t) => !type || t.type === type);
      if (ties.length === 0) reject(`${one.name} and ${other.name} have no tie${type ? ` of type ${type}` : ''}.`);
      const event = await log.append(this.turn, this.actor, 'relation.cancelled', {
        a: one.id,
        b: other.id,
        removed: ties,
      });
      for (const t of ties) await unrelate(db, event.seq, t.src, t.dst, t.type);
      return event;
    });
  }

  /** `note.added`: a quest, promise or plot thread the master must not forget. */
  addNote(text: string): Promise<Outcome> {
    return this.command((tx) => this.appendNote(tx, text));
  }

  /** `note.closed`: the thread is resolved. */
  closeNote(id: number): Promise<Outcome> {
    return this.command(async ({ db, log }) => {
      const note = await log.note(id);
      if (!note) reject(`There is no note #${id}.`);
      if (note.closed) reject(`Note #${id} is already closed.`);
      const event = await log.append(this.turn, this.actor, 'note.closed', { id, text: note.text });
      await db.run('UPDATE notes SET closed_seq = ? WHERE id = ?', [event.seq, id]);
      return event;
    });
  }

  /**
   * `entity.reacted` (ADR 43): the situation is what happened from the entity's side (a line, a
   * grab, a noise); the reaction is its line (words and/or deeds), or "[silence]".
   */
  recordReaction(input: { entity: string; situation: string; reaction: string }): Promise<Outcome> {
    return this.command(async ({ state, log }) => {
      if (!(await state.entity(input.entity))) reject(`There is no ${input.entity}.`);
      return log.append(this.turn, this.actor, 'entity.reacted', input);
    });
  }

  /** Every event of the current turn, in order. */
  turnEvents(): Promise<WorldEvent[]> {
    return this.log.ofTurn(this.turn);
  }

  /** The drafts in the world now (ADR 48). */
  async drafts(): Promise<Entity[]> {
    return (await this.state.allEntities()).filter((e) => e.props.draft === true);
  }

  /** `draft.confirmed`: the draft was used; it stays as an ordinary entity. */
  confirmDraft(id: string): Promise<Outcome> {
    return this.command(async ({ db, state, log }) => {
      const entity = await state.entity(id);
      if (!entity || entity.props.draft !== true) reject(`${id} is not a draft.`);
      const event = await log.append(this.turn, this.actor, 'draft.confirmed', { entity: id, name: entity.name });
      const { draft: _, ...props } = entity.props;
      await putEntity(db, event.seq, { ...entity, props });
      return event;
    });
  }

  /** `draft.withdrawn`: nobody used the draft; it is removed from the world whole. */
  withdrawDraft(id: string): Promise<Outcome> {
    return this.command(async ({ db, state, log }) => {
      const entity = await state.entity(id);
      if (!entity || entity.props.draft !== true) reject(`${id} is not a draft.`);
      const event = await log.append(this.turn, this.actor, 'draft.withdrawn', { entity: id, name: entity.name });
      await forget(db, id);
      return event;
    });
  }

  /** Whether the background work of `turn` has finished (ADR 48). */
  async afterTurnDone(turn: number): Promise<boolean> {
    return !!(await this.db.get('SELECT 1 FROM after_turns WHERE turn = ?', [turn]));
  }

  async markAfterTurnDone(turn: number): Promise<void> {
    await this.db.run('INSERT OR IGNORE INTO after_turns (turn, finished_at) VALUES (?, ?)', [turn, nowIso()]);
  }

  /** `luck.rolled`: journal only — the generator records what the master made of it. */
  recordLuck(roll: EventPayloads['luck.rolled']): Promise<Outcome> {
    return this.command(async ({ log }) => {
      if (!roll.attempt.trim()) reject('Say what the player attempts.');
      return log.append(this.turn, this.actor, 'luck.rolled', { ...roll, attempt: roll.attempt.trim() });
    });
  }

  /** Whether `entity` already reacted this turn (`entity.reacted`). */
  hasReacted(entity: string): Promise<boolean> {
    return this.log.reacted(this.turn, entity);
  }

  /** `note.added` with `entity`: an actor note — what the entity decided, for its next reaction. */
  addActorNote(entity: string, text: string): Promise<Outcome> {
    return this.command(async (tx) => {
      if (!(await tx.state.entity(entity))) reject(`There is no ${entity}.`);
      return this.appendNote(tx, text, entity);
    });
  }

  // --- internals -----------------------------------------------------------------------------

  /** `LOCATED_IN`: the player walks, a being goes anywhere, a thing is put down or moved. */
  private async place(tx: Tx, player: Entity, a: string, b: string): Promise<WorldEvent> {
    const { db, state, log } = tx;
    const thing = await this.find(state, a);
    if (thing.id === player.id) return this.walk(tx, player, b);
    const where = await this.find(state, b);
    if (thing.id === where.id) reject(`${thing.name} cannot be inside itself.`);
    if (isBeing(thing)) {
      if (where.type !== 'LOCATION') reject(`${thing.name} can only go to a location; ${where.name} is not one.`);
      const from = await state.locationOf(thing.id);
      if (from?.id === where.id) reject(`${thing.name} is already in ${where.name}.`);
      const event = await log.append(this.turn, this.actor, 'npc.moved', {
        npc: thing.id,
        from: from?.id ?? '',
        to: where.id,
      });
      await moveTo(db, event.seq, thing.id, where.id);
      return event;
    }
    if (thing.type === 'LOCATION') reject(`${thing.name} is a place; places stay where the story put them.`);
    if (isCarrier(where)) {
      reject(`To give ${thing.name} to ${where.name}, use HAS_ITEM: ${where.name} has ${thing.name}.`);
    }
    if ((await state.inventory(player.id)).some((e) => e.id === thing.id)) {
      const here = await hereOf(state, player);
      const near = [here, ...(await this.reachable(state, player)), ...(await state.inventory(player.id))];
      if (!near.some((e) => e.id === where.id)) {
        reject(`The player can put ${thing.name} down only here (${here.name}) or into something here.`);
      }
      const event = await log.append(this.turn, this.actor, 'item.dropped', { item: thing.id, to: where.id });
      await putIn(db, event.seq, thing.id, where.id);
      return event;
    }
    const [from = ''] = await state.placesOf(thing.id);
    if (from === where.id) reject(`${thing.name} is already in ${where.name}.`);
    const event = await log.append(this.turn, this.actor, 'item.moved', { item: thing.id, from, to: where.id });
    await putIn(db, event.seq, thing.id, where.id);
    return event;
  }

  /** The player goes only along an exit of the current location (`player.moved`). */
  private async walk({ db, state, log }: Tx, player: Entity, target: string): Promise<WorldEvent> {
    const here = await hereOf(state, player);
    if (target === here.id || normalizeName(target) === normalizeName(here.name)) {
      reject(`The player is already in ${here.name}.`);
    }
    const exits = await state.exits(here.id);
    const to = pick(
      resolveName(target, exits.map((exit) => exit.to)),
      target,
      () =>
        `There is no path from ${here.name} to "${target}". Exits: ${
          exits.map((exit) => describeExit(exit)).join('; ') || 'none'
        }.`,
    );
    const event = await log.append(this.turn, this.actor, 'player.moved', { from: here.id, to: to.id });
    await moveTo(db, event.seq, player.id, to.id);
    return event;
  }

  /** `HAS_ITEM`: the player takes, someone else gets a thing; into a place it is `LOCATED_IN`. */
  private async give(tx: Tx, player: Entity, a: string, b: string): Promise<WorldEvent> {
    const { db, state, log } = tx;
    const holder = await this.find(state, a);
    if (holder.id === player.id) return this.take(tx, player, b);
    if (!isBeing(holder)) return this.place(tx, player, b, a); // a place or a container
    const item = await this.find(state, b);
    if (!isPortable(item)) reject(`${item.name} cannot be carried.`);
    if ((await state.inventory(holder.id)).some((e) => e.id === item.id)) {
      reject(`${holder.name} already has ${item.name}.`);
    }
    if ((await state.inventory(player.id)).some((e) => e.id === item.id)) {
      const here = await state.locationOf(player.id);
      if ((await state.locationOf(holder.id))?.id !== here?.id) {
        reject(`${holder.name} is not here to receive ${item.name}.`);
      }
    }
    const [from = ''] = await state.placesOf(item.id);
    const event = await log.append(this.turn, this.actor, 'item.moved', { item: item.id, from, to: holder.id });
    await handTo(db, event.seq, item.id, holder.id);
    return event;
  }

  /**
   * `item.taken`: a portable item that lies here, or inside something that lies here. An item
   * an NPC here holds only after a played-out attempt (ADR 32, 34): this turn already has that
   * NPC's `entity.reacted`, and the master decided the attempt worked.
   */
  private async take({ db, state, log }: Tx, player: Entity, target: string): Promise<WorldEvent> {
    const owned = resolveName(target, await state.inventory(player.id));
    if (owned.kind === 'found') reject(`The player already has ${owned.entity.name}.`);
    const reachable = await this.reachable(state, player);
    const held = await this.heldByNpcsHere(state, player);
    const item = pick(
      resolveName(target, [...reachable, ...held.map((h) => h.item)]),
      target,
      () => `There is no "${target}" here to take.`,
    );
    if (!isPortable(item)) reject(`${item.name} cannot be carried.`);
    if (item.props.fixed === true) reject(`${item.name} is fixed in place and cannot be taken.`);
    const holder = held.find((h) => h.item.id === item.id)?.npc;
    if (holder && !(await log.reacted(this.turn, holder.id))) {
      reject(
        `${holder.name} has ${item.name}. Play out the attempt first: call react_entity for ` +
          `${holder.name}, then take it only if the attempt succeeds.`,
      );
    }
    const [from = ''] = await state.placesOf(item.id);
    const event = await log.append(this.turn, this.actor, 'item.taken', { item: item.id, from });
    await handTo(db, event.seq, item.id, player.id);
    return event;
  }

  /**
   * `relation.set`: writes or rewrites a tie between two entities (anywhere, the player too):
   * allegiance, grudge, debt, trust. An existing tie of this type between them, in either
   * direction, gets the new description; otherwise a new one goes from `a` to `b`.
   */
  private async tie(
    { db, state, log }: Tx,
    player: Entity,
    a: string,
    b: string,
    type: TieType,
    description: string,
  ): Promise<WorldEvent> {
    if (!description.trim()) reject('Describe the tie: how one stands with the other now.');
    const [one, other] = await this.pair(state, a, b);
    const existing = (await this.tiesBetween(db, one.id, other.id)).find((t) => t.type === type);
    const [src, dst] = existing ? [existing.src, existing.dst] : [one.id, other.id];
    const event = await log.append(this.turn, this.actor, 'relation.set', {
      src,
      dst,
      type,
      description: description.trim(),
      before: existing?.description ?? null,
    });
    await relate(db, event.seq, src, dst, type, description.trim());
    return event;
  }

  /** An entity by exact id, else by name: what the player can see first, then the whole world. */
  private async find(state: WorldState, target: string, notFound = () => `There is no "${target}".`): Promise<Entity> {
    const byId = await state.entity(target.trim());
    if (byId) return byId;
    const near = resolveName(target, await state.visible()); // the player is among them
    if (near.kind !== 'none') return pick(near, target, notFound);
    return pick(resolveName(target, await state.allEntities()), target, notFound);
  }

  /** Two different entities, each by id or name. */
  private async pair(state: WorldState, a: string, b: string): Promise<[Entity, Entity]> {
    const one = await this.find(state, a);
    const other = await this.find(state, b);
    if (one.id === other.id) reject(`A tie needs two different parties; both are ${one.name}.`);
    return [one, other];
  }

  /** The ties between two entities, in both directions, as the world has them now. */
  private tiesBetween(db: Db, a: string, b: string): Promise<Tie[]> {
    return db.all<Tie & Record<string, string>>(
      `SELECT src, dst, type, description FROM world_relations
        WHERE ((src = ? AND dst = ?) OR (src = ? AND dst = ?))
          AND type IN (${TIE_TYPES.map(() => '?').join(', ')})
        ORDER BY id`,
      [a, b, b, a, ...TIE_TYPES],
    );
  }

  /** `note.added`: a master's note, or with `entity` an actor note. */
  private async appendNote({ db, log }: Tx, text: string, entity?: string): Promise<WorldEvent> {
    const clean = text.trim();
    if (!clean) reject('The note is empty.');
    const row = await db.get<{ id: number }>('SELECT coalesce(max(id), 0) + 1 AS id FROM notes');
    const id = Number(row?.id ?? 1);
    const payload = entity ? { id, text: clean, entity } : { id, text: clean };
    const event = await log.append(this.turn, this.actor, 'note.added', payload);
    await db.run('INSERT INTO notes (id, text, created_seq, entity_id) VALUES (?, ?, ?, ?)', [
      id,
      clean,
      event.seq,
      entity ?? null,
    ]);
    return event;
  }

  /** Items the player can reach: here, and one level inside containers that are here. */
  private async reachable(state: WorldState, player: Entity): Promise<Entity[]> {
    const here = await state.locationOf(player.id);
    if (!here) return [];
    const direct = await state.contents(here.id);
    const nested = await Promise.all(
      direct.filter(isContainerLike).map((e) => state.contents(e.id)),
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
    const npcs = (await state.contents(here.id)).filter(isBeing);
    const held = await Promise.all(
      npcs.map(async (npc) => (await state.inventory(npc.id)).map((item) => ({ npc, item }))),
    );
    return held.flat();
  }

  /** A command of a started session: refused before the player exists. */
  private command(body: (tx: Tx, player: Entity) => Promise<WorldEvent>): Promise<Outcome> {
    return this.transact(async (tx) => {
      const player = await tx.state.player();
      if (!player) reject('The session has not started.');
      return body(tx, player);
    });
  }

  /** One transaction: the event and its change, or nothing if a rule is broken (`reject`). */
  private async transact(body: (tx: Tx) => Promise<WorldEvent>): Promise<Outcome> {
    let event: WorldEvent;
    try {
      event = await this.db.transaction((db) => body({ db, state: new WorldState(db), log: new EventLog(db) }));
    } catch (error) {
      if (error instanceof Rejected) return fail(error.reason);
      throw error;
    }
    // New or rewritten text: its vectors in the background, after the commit. Without network
    // they stay missing until the next change or the next time the save opens.
    if (event.kind === 'entity.added' || event.kind === 'entity.edited') void this.refreshLore().catch(() => 0);
    return ok(event);
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

/** Where the player is; a session that has not placed them breaks the rule. */
async function hereOf(state: WorldState, player: Entity): Promise<Entity> {
  const here = await state.locationOf(player.id);
  if (!here) reject('The player is nowhere; the session has not placed them.');
  return here;
}

function describeExit(exit: Exit): string {
  return exit.description ? `${exit.to.name} (${exit.description})` : exit.to.name;
}
