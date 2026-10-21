import { afterEach, describe, expect, it } from 'vitest';

import { withDb } from '@/core/storage/db';
import { DEMO_STORY, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import { parseHeroes, readStoryMeta } from './story-meta';

let env: TestEnv | undefined;

afterEach(async () => {
  await env?.cleanup();
  env = undefined;
});

describe('story meta: tagline and heroes (ADR 62)', () => {
  it('reads the demo story’s tagline and heroes', async () => {
    env = await makeTestEnv();
    const meta = await withDb(env.driver, DEMO_STORY, readStoryMeta, { readOnly: true });
    expect(meta.tagline).toBe('Steal the Clockheart before the Admiralty buys it at noon.');
    expect(meta.heroes.map((h) => h.key)).toEqual(['ada', 'elias', 'imogen']);
    expect(meta.heroes[0]).toMatchObject({ name: 'Ada Merrow', sex: 'female', age: 24, race: 'human', blurb: 'Locksmith from Brassworks Alley' });
  });

  it('drops what it cannot use instead of breaking the story', () => {
    expect(parseHeroes(undefined)).toEqual([]);
    expect(parseHeroes('{nope')).toEqual([]);
    expect(parseHeroes('{"key": "a"}')).toEqual([]);
    const heroes = parseHeroes(
      JSON.stringify([
        { key: 'a', name: 'A', description: 'x', age: 0, race: '' },
        { key: 'a', name: 'Again', description: 'y' },
        { key: 'b', name: '', description: 'z' },
        'not a hero',
        { key: 'c', name: 'C', description: 'w', age: 30 },
      ]),
    );
    expect(heroes).toEqual([
      { key: 'a', name: 'A', sex: null, age: null, race: null, blurb: '', description: 'x' },
      { key: 'c', name: 'C', sex: null, age: 30, race: null, blurb: '', description: 'w' },
    ]);
  });
});
