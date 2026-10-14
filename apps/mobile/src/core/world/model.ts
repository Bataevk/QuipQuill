/** World model: what the game reads from `state_*` and writes to `events` (data-model.md). */

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

/** 'player:<id>' | 'gm' | 'npc:<id>' | 'system' */
export type Actor = string;

/** Payload of each event kind. One event = one transaction with its state change. */
export interface EventPayloads {
  'session.started': { story_id: string; character_id: string | null; player: string };
  'character.adapted': { original: Props; adapted: Props; choice: string };
  'player.moved': { from: string; to: string };
  'item.taken': { item: string; from: string };
  'item.dropped': { item: string; to: string };
  'entity.added': { entity: string; relation: { type: string; target: string } };
  'entity.edited': { id: string; before: EntityPatch; after: EntityPatch };
  'player.updated': { before: string; after: string };
  'note.added': { id: number; text: string };
  'note.closed': { id: number; text: string };
  /** An NPC reacted to something (a line, an action, an event) at the master's call (ADR 33). */
  'npc.reacted': { npc: string; situation: string; reaction: string };
}

export type EventKind = keyof EventPayloads;

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

export interface EntityPatch {
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
