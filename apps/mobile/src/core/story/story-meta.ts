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
  /** How lucky this world is, 0..1 (optional `meta.luck`, ADR 47): 0.5 is fair, below 0.25 brings mishaps. */
  luck: number;
  /** One line under the title on the cover (optional `meta.tagline`). */
  tagline: string;
  /** The author's ready-made player characters (optional `meta.heroes`, ADR 62). */
  heroes: StoryHero[];
}

/** A hero written by the story's author (ADR 62); `key` is stable across rebuilds. */
export interface StoryHero {
  key: string;
  name: string;
  sex: string | null;
  age: number | null;
  race: string | null;
  /** A few words for the picker: "Locksmith from Brassworks Alley". */
  blurb: string;
  description: string;
}

/** A fair world, when the story does not say. */
export const DEFAULT_LUCK = 0.5;

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
    luck: parseLuck(meta.get('luck')),
    tagline: text('tagline'),
    heroes: parseHeroes(meta.get('heroes')),
  };
}

/**
 * `meta.heroes` as sheets; anything unreadable is no heroes rather than a broken story (the
 * builder validates them, ADR 62). A sheet without a key, a name or a description is dropped.
 */
export function parseHeroes(value: string | undefined): StoryHero[] {
  let raw: unknown;
  try {
    raw = value?.trim() ? JSON.parse(value) : [];
  } catch {
    return [];
  }
  if (!Array.isArray(raw)) return [];
  const str = (v: unknown) => (typeof v === 'string' && v.trim() ? v.trim() : null);
  const heroes: StoryHero[] = [];
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue;
    const hero = item as Record<string, unknown>;
    const key = str(hero.key);
    const name = str(hero.name);
    const description = str(hero.description);
    if (!key || !name || !description || heroes.some((h) => h.key === key)) continue;
    const age = typeof hero.age === 'number' && Number.isInteger(hero.age) && hero.age > 0 ? hero.age : null;
    heroes.push({ key, name, sex: str(hero.sex), age, race: str(hero.race), blurb: str(hero.blurb) ?? '', description });
  }
  return heroes;
}

/** `meta.luck` as a number in 0..1; missing or unreadable → DEFAULT_LUCK. */
export function parseLuck(value: string | undefined): number {
  const luck = value === undefined || !value.trim() ? NaN : Number(value);
  return Number.isFinite(luck) ? Math.min(1, Math.max(0, luck)) : DEFAULT_LUCK;
}
