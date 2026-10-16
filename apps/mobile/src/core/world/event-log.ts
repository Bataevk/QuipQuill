import type { Db } from '../storage/db';
import { nowIso } from '../util/ids';
import type { Actor, EventKind, EventPayloads, Note, WorldEvent } from './model';

type EventRow = { seq: number; turn: number; actor: string; kind: string; payload: string; ts: string };

/** The append-only journal (`events`) and the notes (`notes`): the master's and actors'. */
export class EventLog {
  private readonly db: Db;

  constructor(db: Db) {
    this.db = db;
  }

  /** Appends an event and returns it with its `seq`. Call inside the state change's transaction. */
  async append<K extends EventKind>(
    turn: number,
    actor: Actor,
    kind: K,
    payload: EventPayloads[K],
  ): Promise<WorldEvent> {
    const ts = nowIso();
    const { lastInsertRowId } = await this.db.run(
      'INSERT INTO events (turn, actor, kind, payload, ts) VALUES (?, ?, ?, ?, ?)',
      [turn, actor, kind, JSON.stringify(payload), ts],
    );
    return { seq: lastInsertRowId, turn, actor, kind, payload, ts } as WorldEvent;
  }

  /** The last `limit` events, oldest first (the master sees the last 5, data-model.md). */
  async recent(limit: number): Promise<WorldEvent[]> {
    const rows = await this.db.all<EventRow>(
      'SELECT * FROM (SELECT * FROM events ORDER BY seq DESC LIMIT ?) ORDER BY seq',
      [limit],
    );
    return rows.map(toEvent);
  }

  /** Every event of one turn, in order. */
  async ofTurn(turn: number): Promise<WorldEvent[]> {
    const rows = await this.db.all<EventRow>('SELECT * FROM events WHERE turn = ? ORDER BY seq', [turn]);
    return rows.map(toEvent);
  }

  async count(kind?: EventKind): Promise<number> {
    const row = kind
      ? await this.db.get<{ n: number }>('SELECT count(*) AS n FROM events WHERE kind = ?', [kind])
      : await this.db.get<{ n: number }>('SELECT count(*) AS n FROM events');
    return Number(row?.n ?? 0);
  }

  /** Whether `entity` already reacted (`entity.reacted`) during `turn`. */
  async reacted(turn: number, entity: string): Promise<boolean> {
    const row = await this.db.get<{ n: number }>(
      `SELECT count(*) AS n FROM events
        WHERE turn = ? AND kind = 'entity.reacted' AND json_extract(payload, '$.entity') = ?`,
      [turn, entity],
    );
    return Number(row?.n ?? 0) > 0;
  }

  async lastTurn(): Promise<number> {
    const row = await this.db.get<{ turn: number | null }>('SELECT max(turn) AS turn FROM events');
    return Number(row?.turn ?? 0);
  }

  /** The master's open notes (actor notes are not his). */
  async openNotes(): Promise<Note[]> {
    const rows = await this.db.all<{ id: number; text: string; created_seq: number }>(
      'SELECT id, text, created_seq FROM notes WHERE closed_seq IS NULL AND entity_id IS NULL ORDER BY id',
    );
    return rows.map(toNote);
  }

  /** The last `limit` actor notes of `entity`, oldest first. */
  async actorNotes(entity: string, limit: number): Promise<Note[]> {
    const rows = await this.db.all<{ id: number; text: string; created_seq: number }>(
      `SELECT * FROM (SELECT id, text, created_seq FROM notes WHERE entity_id = ? ORDER BY id DESC LIMIT ?)
        ORDER BY id`,
      [entity, limit],
    );
    return rows.map(toNote);
  }

  /** A master's note by id. */
  async note(id: number): Promise<(Note & { closed: boolean }) | undefined> {
    const row = await this.db.get<{ id: number; text: string; created_seq: number; closed_seq: number | null }>(
      'SELECT * FROM notes WHERE id = ? AND entity_id IS NULL',
      [id],
    );
    return (
      row && {
        id: Number(row.id),
        text: row.text,
        createdSeq: Number(row.created_seq),
        closed: row.closed_seq !== null,
      }
    );
  }
}

function toNote(row: { id: number; text: string; created_seq: number }): Note {
  return { id: Number(row.id), text: row.text, createdSeq: Number(row.created_seq) };
}

function toEvent(row: EventRow): WorldEvent {
  return {
    seq: Number(row.seq),
    turn: Number(row.turn),
    actor: row.actor,
    kind: row.kind as EventKind,
    payload: JSON.parse(row.payload),
    ts: row.ts,
  } as WorldEvent;
}
