import type { Db } from '../storage/db';
import { nowIso } from '../util/ids';

/** 'player' | 'gm' | 'npc:<id>' | 'system' */
export type Speaker = string;

export interface TranscriptLine {
  seq: number;
  speaker: Speaker;
  text: string;
  extra: Record<string, unknown>;
  ts: string;
}

/**
 * What the player saw (`transcript`, data-model.md): only appended, never summarized — unlike
 * the models' working memory in `model_messages`. The game screen restores from it (5.3).
 */
export class TranscriptStore {
  private readonly db: Db;

  constructor(db: Db) {
    this.db = db;
  }

  async append(speaker: Speaker, text: string, extra: Record<string, unknown> = {}): Promise<TranscriptLine> {
    const ts = nowIso();
    const { lastInsertRowId } = await this.db.run(
      'INSERT INTO transcript (speaker, text, extra, ts) VALUES (?, ?, ?, ?)',
      [speaker, text, JSON.stringify(extra), ts],
    );
    return { seq: lastInsertRowId, speaker, text, extra, ts };
  }

  /** The last `limit` lines (all by default), oldest first. */
  async lines(limit = -1): Promise<TranscriptLine[]> {
    const rows = await this.db.all<{ seq: number; speaker: string; text: string; extra: string; ts: string }>(
      'SELECT * FROM (SELECT * FROM transcript ORDER BY seq DESC LIMIT ?) ORDER BY seq',
      [limit],
    );
    return rows.map((row) => ({
      seq: Number(row.seq),
      speaker: row.speaker,
      text: row.text,
      extra: JSON.parse(row.extra) as Record<string, unknown>,
      ts: row.ts,
    }));
  }
}
