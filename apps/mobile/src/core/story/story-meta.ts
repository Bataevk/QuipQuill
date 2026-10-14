import type { Db } from '../storage/db';

/** The .qqstory schema this app understands (schema/story.sql, `meta.schema_version`). */
export const STORY_SCHEMA_VERSION = '1';

/** Keys the game cannot work without. Authored texts may still be drafts. */
const REQUIRED = ['schema_version', 'story_id', 'title', 'start_location', 'embed_model', 'embed_dim'];

export interface StoryMeta {
  storyId: string;
  title: string;
  language: string;
  startLocation: string;
  worldSummary: string;
  worldConstraints: string;
  style: string;
  embedModel: string;
  embedDim: number;
  builtAt: string;
}

/** The file is not a story this app can play; `message` is safe to show to the player. */
export class StoryFormatError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'StoryFormatError';
  }
}

export async function readStoryMeta(db: Db): Promise<StoryMeta> {
  let rows: { key: string; value: string }[];
  try {
    rows = await db.all<{ key: string; value: string }>('SELECT key, value FROM meta');
  } catch (error) {
    throw new StoryFormatError('This file is not a QuipQuill story.', { cause: error });
  }
  const meta = new Map(rows.map((row) => [row.key, String(row.value)]));
  const missing = REQUIRED.filter((key) => !meta.get(key)?.trim());
  if (missing.length > 0) {
    throw new StoryFormatError(`The story file is incomplete (missing: ${missing.join(', ')}).`);
  }
  if (meta.get('schema_version') !== STORY_SCHEMA_VERSION) {
    throw new StoryFormatError(
      `Story format v${meta.get('schema_version')} is not supported (this app reads v${STORY_SCHEMA_VERSION}).`,
    );
  }
  const embedDim = Number(meta.get('embed_dim'));
  if (!Number.isInteger(embedDim) || embedDim <= 0) {
    throw new StoryFormatError(`The story file has a bad embed_dim: ${meta.get('embed_dim')}.`);
  }
  const text = (key: string) => meta.get(key) ?? '';
  return {
    storyId: text('story_id'),
    title: text('title'),
    language: text('language') || 'en',
    startLocation: text('start_location'),
    worldSummary: text('world_summary'),
    worldConstraints: text('world_constraints'),
    style: text('style'),
    embedModel: text('embed_model'),
    embedDim,
    builtAt: text('built_at'),
  };
}
