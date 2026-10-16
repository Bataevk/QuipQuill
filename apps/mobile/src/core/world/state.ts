import type { Db } from '../storage/db';
import type { Entity, EntityType, Exit, Props } from './model';
import { normalizeName, similarity } from './names';

type EntityRow = {
  id: string;
  type: string;
  subtype: string | null;
  name: string;
  description: string;
  props: string;
};

const ENTITY_COLUMNS = 'e.id, e.type, e.subtype, e.name, e.description, e.props';
const FUZZY_THRESHOLD = 0.8;

export interface Tie {
  other: Entity;
  type: string;
  description: string;
}

export type Resolution =
  | { kind: 'found'; entity: Entity }
  | { kind: 'ambiguous'; candidates: Entity[] }
  | { kind: 'none' };

/**
 * Read-only queries over the playthrough's world: the `world_*` views, where the dynamic graph
 * overrides the story's (ADR 42). Relations follow the
 * story's conventions: `X LOCATED_IN place|container`, `holder HAS_ITEM item` (inventory, or a
 * container's contents), `place LEADS_TO place`.
 */
export class WorldState {
  private readonly db: Db;

  constructor(db: Db) {
    this.db = db;
  }

  async entity(id: string): Promise<Entity | undefined> {
    const row = await this.db.get<EntityRow>(
      `SELECT ${ENTITY_COLUMNS} FROM world_entities e WHERE e.id = ?`,
      [id],
    );
    return row && toEntity(row);
  }

  /** The player's entity (single-player prototype: exactly one PLAYER after session start). */
  async player(): Promise<Entity | undefined> {
    const row = await this.db.get<EntityRow>(
      `SELECT ${ENTITY_COLUMNS} FROM world_entities e WHERE e.type = 'PLAYER' LIMIT 1`,
    );
    return row && toEntity(row);
  }

  /** Where an entity is (`LOCATED_IN` target): a location for the player and NPCs. */
  async locationOf(id: string): Promise<Entity | undefined> {
    return this.one(
      `SELECT ${ENTITY_COLUMNS} FROM world_relations r JOIN world_entities e ON e.id = r.dst
        WHERE r.src = ? AND r.type = 'LOCATED_IN' LIMIT 1`,
      [id],
    );
  }

  async exits(locationId: string): Promise<Exit[]> {
    const rows = await this.db.all<EntityRow & { path: string }>(
      `SELECT ${ENTITY_COLUMNS}, r.description AS path
         FROM world_relations r JOIN world_entities e ON e.id = r.dst
        WHERE r.src = ? AND r.type = 'LEADS_TO' ORDER BY e.name`,
      [locationId],
    );
    return rows.map((row) => ({ to: toEntity(row), description: row.path }));
  }

  /** What is directly in a place or container (`LOCATED_IN` it, or held by it), minus players. */
  async contents(id: string): Promise<Entity[]> {
    return this.many(
      `SELECT ${ENTITY_COLUMNS} FROM world_entities e
        WHERE e.type <> 'PLAYER' AND e.id IN (
          SELECT src FROM world_relations WHERE dst = ? AND type = 'LOCATED_IN'
          UNION SELECT dst FROM world_relations WHERE src = ? AND type = 'HAS_ITEM')
        ORDER BY e.type, e.name`,
      [id, id],
    );
  }

  async inventory(holderId: string): Promise<Entity[]> {
    return this.many(
      `SELECT ${ENTITY_COLUMNS} FROM world_relations r JOIN world_entities e ON e.id = r.dst
        WHERE r.src = ? AND r.type = 'HAS_ITEM' ORDER BY e.name`,
      [holderId],
    );
  }

  /** Who or what holds an item or has it inside: ids from `LOCATED_IN` and `HAS_ITEM`. */
  async placesOf(itemId: string): Promise<string[]> {
    const rows = await this.db.all<{ id: string }>(
      `SELECT dst AS id FROM world_relations WHERE src = ? AND type = 'LOCATED_IN'
       UNION SELECT src AS id FROM world_relations WHERE dst = ? AND type = 'HAS_ITEM'`,
      [itemId, itemId],
    );
    return rows.map((row) => row.id);
  }

  /**
   * What an entity is tied to besides place and possessions (INTERACTS_WITH, HAS_ATTRIBUTE…),
   * both directions, with the story's description of the tie — an NPC's acquaintances,
   * grudges and interests.
   */
  async ties(id: string): Promise<Tie[]> {
    const rows = await this.db.all<EntityRow & { tie: string; about: string }>(
      `SELECT ${ENTITY_COLUMNS}, r.type AS tie, r.description AS about
         FROM world_relations r JOIN world_entities e ON e.id = CASE WHEN r.src = ? THEN r.dst ELSE r.src END
        WHERE (r.src = ? OR r.dst = ?) AND r.type NOT IN ('LEADS_TO', 'LOCATED_IN', 'HAS_ITEM')
        ORDER BY r.id`,
      [id, id, id],
    );
    return rows.map((row) => ({ other: toEntity(row), type: row.tie, description: row.about }));
  }

  async allEntities(): Promise<Entity[]> {
    return this.many(`SELECT ${ENTITY_COLUMNS} FROM world_entities e ORDER BY e.name`);
  }

  /**
   * What the player can see or name right now: the player, the location, its exits, what is
   * here (and one level inside containers here), the inventory.
   */
  async visible(): Promise<Entity[]> {
    const player = await this.player();
    if (!player) return [];
    const location = await this.locationOf(player.id);
    const seen = new Map<string, Entity>([[player.id, player]]);
    const add = (entities: Entity[]) => entities.forEach((e) => seen.set(e.id, e));
    if (location) {
      add([location]);
      add((await this.exits(location.id)).map((exit) => exit.to));
      const here = await this.contents(location.id);
      add(here);
      for (const thing of here.filter(isContainerLike)) add(await this.contents(thing.id));
    }
    add(await this.inventory(player.id));
    return [...seen.values()];
  }

  /**
   * Finds one entity by id or name among `among`: id → exact name → name contains / is
   * contained → fuzzy. Several equally good matches are returned as candidates.
   */
  resolve(nameOrId: string, among: Entity[]): Resolution {
    const byId = among.find((e) => e.id === nameOrId);
    if (byId) return { kind: 'found', entity: byId };
    const key = normalizeName(nameOrId);
    if (!key) return { kind: 'none' };
    const named = among.map((entity) => ({ entity, name: normalizeName(entity.name) }));
    const stages = [
      named.filter((n) => n.name === key),
      named.filter((n) => n.name.includes(key) || key.includes(n.name)),
      named.filter((n) => similarity(n.name, key) >= FUZZY_THRESHOLD),
    ];
    for (const matches of stages) {
      if (matches.length === 1) return { kind: 'found', entity: matches[0].entity };
      if (matches.length > 1) return { kind: 'ambiguous', candidates: matches.map((m) => m.entity) };
    }
    return { kind: 'none' };
  }

  private async one(sql: string, params: string[]): Promise<Entity | undefined> {
    const row = await this.db.get<EntityRow>(sql, params);
    return row && toEntity(row);
  }

  private async many(sql: string, params: string[] = []): Promise<Entity[]> {
    return (await this.db.all<EntityRow>(sql, params)).map(toEntity);
  }
}

function isContainerLike(entity: Entity): boolean {
  return entity.type === 'ITEM' || entity.type === 'DOCUMENT' || entity.type === 'OTHER';
}

function toEntity(row: EntityRow): Entity {
  let props: Props = {};
  try {
    props = JSON.parse(row.props) as Props;
  } catch {
    // CHECK (json_valid) guarantees valid JSON; keep {} if it ever is not an object.
  }
  return {
    id: row.id,
    type: row.type as EntityType,
    subtype: row.subtype,
    name: row.name,
    description: row.description,
    props,
  };
}
