/** World model: what the game reads from `world_*` and writes to `dyn_*` and `events` (data-model.md). */

export type EntityType =
  | 'LOCATION'
  | 'ITEM'
  | 'NPC'
  | 'CREATURE'
  | 'ORGANIZATION'
  | 'DOCUMENT'
  | 'OTHER'
  | 'PLAYER';

export type Props = Record<string, unknown>;

/** Entity types the master and the generator may create; the map itself comes only from the story. */
export const ADDABLE_TYPES = ['ITEM', 'DOCUMENT', 'NPC', 'CREATURE', 'OTHER'] as const satisfies readonly EntityType[];
export type AddableType = (typeof ADDABLE_TYPES)[number];
/** What the Guard may add as a draft before the master: people and fixtures, never items (ADR 48). */
export const DRAFT_TYPES = ['NPC', 'CREATURE', 'OTHER'] as const satisfies readonly AddableType[];
const PORTABLE_TYPES: readonly EntityType[] = ['ITEM', 'DOCUMENT'];
const BEING_TYPES: readonly EntityType[] = ['NPC', 'CREATURE'];
const CONTAINER_LIKE_TYPES: readonly EntityType[] = ['ITEM', 'DOCUMENT', 'OTHER'];

export const isAddableType = (type: string): type is AddableType => (ADDABLE_TYPES as readonly string[]).includes(type);
/** Can be carried: items and documents. */
export const isPortable = (entity: Pick<Entity, 'type'>) => PORTABLE_TYPES.includes(entity.type);
/** People and creatures: they go to any location, react and carry things. */
export const isBeing = (entity: Pick<Entity, 'type'>) => BEING_TYPES.includes(entity.type);
/** Those who carry things (`HAS_ITEM` = inventory) rather than contain them: beings and the player. */
export const isCarrier = (entity: Pick<Entity, 'type'>) => isBeing(entity) || entity.type === 'PLAYER';
/** May have things inside (a chest, a desk, a letter in a folder): one level of it is in reach. */
export const isContainerLike = (entity: Pick<Entity, 'type'>) => CONTAINER_LIKE_TYPES.includes(entity.type);

/** Type names as the model writes them in tools and answers: 'item', 'npc'… */
export function lowerTypes<const T extends readonly EntityType[]>(types: T) {
  return types.map((type) => type.toLowerCase()) as { -readonly [K in keyof T]: Lowercase<T[K]> };
}

export interface Entity {
  id: string;
  type: EntityType;
  subtype: string | null;
  name: string;
  description: string;
  props: Props;
}

export interface Exit {
  to: Entity;
  /** Direction or path as the story says it ("north, through the oak door"); may be empty. */
  description: string;
}

/** The player character's entity id (ADR 43): one player per playthrough. */
export const PLAYER_ID = 'player';

/**
 * Who an event is journaled for: the master, the background generator, the Guard (drafts), the
 * system (session start, entity memory); `npc:<id>` comes only from saves of stage 4.3.
 */
export type Actor = 'gm' | 'generator' | 'guard' | 'system' | `npc:${string}`;

/** Payload of each event kind. One event = one transaction with its state change. */
export interface EventPayloads {
  'session.started': { story_id: string; character_id: string | null; player: string };
  'character.adapted': { original: Props; adapted: Props; choice: string };
  'player.moved': { from: string; to: string };
  'item.taken': { item: string; from: string };
  'item.dropped': { item: string; to: string };
  'entity.added': { entity: string; relation: { type: RelationType; target: string } };
  'entity.edited': { id: string; before: EntityPatch; after: EntityPatch };
  /** A master's note, or with `entity` an actor's note: what that entity decided (ADR 43). */
  'note.added': { id: number; text: string; entity?: string };
  'note.closed': { id: number; text: string };
  /** A thing the master moved or handed over, not by the player's own hand (ADR 43). */
  'item.moved': { item: string; from: string; to: string };
  /** A person or creature went to another location, in sight or not (ADR 42). */
  'npc.moved': { npc: string; from: string; to: string };
  /** A tie was written or rewritten; `before` is its old description, null if it is new. */
  'relation.set': { src: string; dst: string; type: TieType; description: string; before: string | null };
  /** The ties between two entities that are no longer true (story ones get CANCEL, ADR 42). */
  'relation.cancelled': { a: string; b: string; removed: Tie[] };
  /**
   * An entity reacted to something (a line, an action, an event) at the master's call (ADR 33,
   * 43, 45): `reaction` is its line (words and/or deeds), or "[silence]".
   */
  'entity.reacted': { entity: string; situation: string; reaction: string };
  /** A draft the Guard added before the master (ADR 48) was used this turn and stays. */
  'draft.confirmed': { entity: string; name: string };
  /** A draft nobody used this turn is gone from the world. */
  'draft.withdrawn': { entity: string; name: string };
  /** A roll of the player's luck at the master's call (ADR 47); the master narrates the outcome. */
  'luck.rolled': {
    attempt: string;
    likelihood: string;
    stakes: string;
    luck: number;
    chance: number;
    roll: number;
    outcome: 'success' | 'failure' | 'mishap';
  };
}

export type EventKind = keyof EventPayloads;

/**
 * The entity ids an event names (prefixed ids only, so not the player), in a fixed order: the
 * generator takes the people of a turn in this order.
 */
export function eventEntityIds(event: WorldEvent): string[] {
  const p = event.payload as Record<string, unknown>;
  return ['npc', 'id', 'entity', 'src', 'dst', 'a', 'b', 'item', 'from', 'to']
    .map((key) => p[key])
    .filter((v): v is string => typeof v === 'string' && v.includes(':'));
}

export type WorldEvent = {
  [K in EventKind]: {
    seq: number;
    turn: number;
    actor: Actor;
    kind: K;
    payload: EventPayloads[K];
    ts: string;
  };
}[EventKind];

/** Where things are and how places connect: `LOCATED_IN`, `HAS_ITEM`; `LEADS_TO` only from the story. */
export const PLACEMENT_TYPES = ['LOCATED_IN', 'HAS_ITEM', 'LEADS_TO'] as const;
/**
 * Relation types that are ties: who knows, serves, owes, hates whom (set_relation,
 * cancel_relation, `state.ties`). The builder writes no other types besides PLACEMENT_TYPES.
 */
export const TIE_TYPES = ['INTERACTS_WITH', 'HAS_ATTRIBUTE'] as const;
export type TieType = (typeof TIE_TYPES)[number];
export type RelationType = (typeof PLACEMENT_TYPES)[number] | TieType;

export const isTieType = (type: string): type is TieType => (TIE_TYPES as readonly string[]).includes(type);

/** A tie between two entities, as the world has it (`relation.cancelled` keeps the ones it removed). */
export interface Tie {
  src: string;
  dst: string;
  type: TieType;
  description: string;
}

export interface EntityPatch {
  name?: string;
  description?: string;
  props?: Props;
}

export interface Note {
  id: number;
  text: string;
  createdSeq: number;
}

/**
 * Outcome of a world command. A broken rule (no path there, the item is elsewhere) is a normal
 * outcome, not an exception: the master's tools return `error` to the model as text.
 */
export type Outcome<T = WorldEvent> = { ok: true; value: T } | { ok: false; error: string };

export const ok = <T>(value: T): Outcome<T> => ({ ok: true, value });
export const fail = <T = never>(error: string): Outcome<T> => ({ ok: false, error });
