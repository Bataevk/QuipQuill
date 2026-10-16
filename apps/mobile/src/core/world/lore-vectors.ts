import { blob, type Db } from '../storage/db';
import { rankByCosine, toEntry } from '../vector/brute-force-index';
import { toVector, type EmbeddingKind, type VectorHit, type VectorIndex } from '../vector/vector-index';

/** Embeds texts with the story's `meta.embed_model` (CloudEmbeddings in the app). */
export type Embed = (texts: string[]) => Promise<Float32Array[]>;

type Texts = Partial<Record<EmbeddingKind, string>>;
const KINDS: readonly EmbeddingKind[] = ['name', 'desc'];

/** What gets embedded, as builder does it: the name, and "name: description" when described. */
export function embeddingTexts(entity: { name: string; description: string }): Texts {
  return {
    name: entity.name,
    ...(entity.description.trim() ? { desc: `${entity.name}: ${entity.description}` } : {}),
  };
}

type OverrideRow = {
  id: string;
  name: string;
  description: string;
  story_name: string | null;
  story_description: string | null;
};

/** Per (entity, kind): which story vectors are out of date, which texts need a dynamic one. */
interface Plan {
  /** `id/kind` of story vectors whose text the dynamic graph has rewritten. */
  hidden: Set<string>;
  /** `id/kind` → the text a dynamic vector must have been computed from to count. */
  wanted: Map<string, { entityId: string; kind: EmbeddingKind; text: string }>;
}

const key = (entityId: string, kind: string) => `${entityId}/${kind}`;

/**
 * Lore search over both graphs (ADR 42): the story's index, minus the vectors of entities the
 * playthrough has renamed or rewritten, plus `dyn_embeddings` scanned in JS (they are few).
 * A dynamic vector counts only while its `text` is the entity's text now, so a stale one is
 * never found by the old meaning; `refresh()` computes what is missing, outside any transaction.
 */
export class LoreVectors {
  private readonly db: Db;
  private readonly story: VectorIndex | undefined;
  private readonly embed: Embed | undefined;
  private refreshing: Promise<number> = Promise.resolve(0);

  constructor(db: Db, story: VectorIndex | undefined, embed: Embed | undefined) {
    this.db = db;
    this.story = story;
    this.embed = embed;
  }

  get available(): boolean {
    return this.story !== undefined;
  }

  async search(query: Float32Array, limit: number): Promise<VectorHit[]> {
    if (limit <= 0) return [];
    const plan = await this.plan();
    // Ask the story index for enough extra hits to make up for the hidden ones.
    const story = this.story ? await this.story.search(query, limit + plan.hidden.size) : [];
    const rows = await this.db.all<{ entity_id: string; kind: string; text: string; vector: Uint8Array }>(
      'SELECT entity_id, kind, text, vector FROM dyn_embeddings',
    );
    const current = rows
      .filter((row) => plan.wanted.get(key(row.entity_id, row.kind))?.text === row.text)
      .map((row) => toEntry(row.entity_id, row.kind as EmbeddingKind, toVector(row.vector)));
    return [
      ...story.filter((hit) => !plan.hidden.has(key(hit.entityId, hit.kind))),
      ...rankByCosine(current, query, limit),
    ]
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  /**
   * Computes the dynamic vectors that are missing or stale; runs one at a time. Resolves with
   * how many were written; rejects if embedding fails (no network) — they stay missing until
   * the next call, and search meanwhile skips the rewritten texts.
   */
  refresh(): Promise<number> {
    const run = this.refreshing.catch(() => 0).then(() => this.refreshNow());
    this.refreshing = run;
    return run;
  }

  private async refreshNow(): Promise<number> {
    if (!this.embed) return 0;
    const { wanted } = await this.plan();
    const have = new Map(
      (await this.db.all<{ entity_id: string; kind: string; text: string }>(
        'SELECT entity_id, kind, text FROM dyn_embeddings',
      )).map((row) => [key(row.entity_id, row.kind), row.text]),
    );
    const missing = [...wanted.entries()].filter(([k, want]) => have.get(k) !== want.text).map(([, w]) => w);
    if (missing.length === 0) return 0;
    const vectors = await this.embed(missing.map((m) => m.text));
    // Single statements, no transaction: a row is self-checking through its text, so one that
    // is outdated by the time it lands is simply skipped by search and redone next time.
    for (const [i, m] of missing.entries()) {
      await this.db.run(
        'INSERT OR REPLACE INTO dyn_embeddings (entity_id, kind, text, vector) VALUES (?, ?, ?, ?)',
        [m.entityId, m.kind, m.text, blob(vectors[i])],
      );
    }
    return missing.length;
  }

  private async plan(): Promise<Plan> {
    const rows = await this.db.all<OverrideRow>(
      `SELECT d.id, d.name, d.description, e.name AS story_name, e.description AS story_description
         FROM dyn_entities d LEFT JOIN entities e ON e.id = d.id
        WHERE d.type <> 'PLAYER'`,
    );
    const plan: Plan = { hidden: new Set(), wanted: new Map() };
    for (const row of rows) {
      const now = embeddingTexts(row);
      const story: Texts =
        row.story_name === null
          ? {}
          : embeddingTexts({ name: row.story_name, description: row.story_description ?? '' });
      for (const kind of KINDS) {
        if (now[kind] === story[kind]) continue; // the story's own vector still fits
        if (story[kind] !== undefined) plan.hidden.add(key(row.id, kind));
        const text = now[kind];
        if (text !== undefined) plan.wanted.set(key(row.id, kind), { entityId: row.id, kind, text });
      }
    }
    return plan;
  }
}
