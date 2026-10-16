import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { Db } from '@/core/storage/db';
import { openVectorIndex } from '@/core/vector/open-index';
import { toVector } from '@/core/vector/vector-index';
import { putEntity } from '@/core/world/dynamic-graph';
import type { Outcome } from '@/core/world/model';
import { World } from '@/core/world/world';
import { DEMO_STORY, DEMO_STORY_ID, fakeEmbedding, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

const TAVERN = 'loc:rusty-gear-tavern';
const MAGS = 'npc:mags-holloway';
const PIP = 'npc:pip-quillon';
const CHARACTER = { id: 'c1', name: 'Aria', description: 'A wandering cartographer.' };
const NEW_MAGS = 'Gave up on the heist after the player refused to join; now tends a flower stall.';

let env: TestEnv;
let db: Db;
let world: World;
/** The fake embedding service: off simulates no network. */
let online: boolean;
let embedded: string[];

beforeEach(async () => {
  env = await makeTestEnv();
  await env.library.importer.import(DEMO_STORY);
  const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 't' });
  db = await env.library.saves.open(save.id);
  online = true;
  embedded = [];
  world = await openWorld();
  world.beginTurn();
  expectOk(await world.startSession({ storyId: DEMO_STORY_ID, character: CHARACTER, startLocation: TAVERN }));
});

afterEach(async () => {
  await world?.refreshLore().catch(() => 0); // let background work finish before closing
  await db?.close();
  await env?.cleanup();
});

async function openWorld(): Promise<World> {
  return World.open(db, {
    vectors: await openVectorIndex(db),
    embed: async (texts) => {
      if (!online) throw new Error('network is down');
      embedded.push(...texts);
      return texts.map((text) => fakeEmbedding(text));
    },
  });
}

function expectOk<T>(outcome: Outcome<T>): T {
  if (!outcome.ok) throw new Error(`expected ok, got: ${outcome.error}`);
  return outcome.value;
}

async function storyVector(id: string, kind: 'name' | 'desc'): Promise<Float32Array> {
  const row = await db.get<{ vector: Uint8Array }>(
    'SELECT vector FROM embeddings WHERE entity_id = ? AND kind = ?',
    [id, kind],
  );
  return toVector(row!.vector);
}

/** Best score of an entity among the top 10 lore hits for `query`, or undefined. */
async function scoreOf(id: string, query: Float32Array): Promise<number | undefined> {
  const hits = expectOk(await world.searchLore(query, 10));
  return hits.find((hit) => hit.entity.id === id)?.score;
}

const top = async (query: Float32Array) => expectOk(await world.searchLore(query, 3))[0]?.entity.id;

describe('lore search over the dynamic graph', () => {
  it('finds a rewritten NPC by the new meaning and not by the old one', async () => {
    const oldMeaning = await storyVector(MAGS, 'desc');
    expect(await top(oldMeaning)).toBe(MAGS);

    expectOk(await world.setEntity({ id: MAGS, description: NEW_MAGS }));
    await world.refreshLore();

    expect(await top(oldMeaning)).not.toBe(MAGS);
    expect(await scoreOf(MAGS, oldMeaning) ?? 0).toBeLessThan(0.9); // her name still matches a little
    expect(await top(fakeEmbedding(`Mags Holloway: ${NEW_MAGS}`))).toBe(MAGS);
    // the name did not change: only the description gets a vector of its own
    expect(await db.all('SELECT entity_id, kind FROM dyn_embeddings')).toEqual([
      { entity_id: MAGS, kind: 'desc' },
    ]);
  });

  it('computes the vectors in the background after the change', async () => {
    expectOk(await world.setEntity({ id: MAGS, description: NEW_MAGS }));
    await vi.waitFor(async () => {
      expect(await db.all('SELECT kind FROM dyn_embeddings')).toEqual([{ kind: 'desc' }]);
    });
    expect(embedded).toEqual([`Mags Holloway: ${NEW_MAGS}`]);
  });

  it('works without network: the rewritten text is skipped until its vector is computed', async () => {
    online = false;
    const oldMeaning = await storyVector(MAGS, 'desc');
    expectOk(await world.setEntity({ id: MAGS, description: NEW_MAGS }));
    await expect(world.refreshLore()).rejects.toThrow('network is down');

    expect(await top(oldMeaning)).not.toBe(MAGS);
    expect(await top(await storyVector(PIP, 'desc'))).toBe(PIP); // the rest of the lore is there
    expect(await scoreOf(MAGS, fakeEmbedding(`Mags Holloway: ${NEW_MAGS}`)) ?? 0).toBeLessThan(0.9);

    online = true; // the next refresh (next change or the next time the save opens) catches up
    expect(await world.refreshLore()).toBe(1);
    expect(await top(fakeEmbedding(`Mags Holloway: ${NEW_MAGS}`))).toBe(MAGS);
  });

  it('does not use a vector of an earlier rewrite', async () => {
    expectOk(await world.setEntity({ id: MAGS, description: 'First rewrite.' }));
    await world.refreshLore();
    online = false;
    expectOk(await world.setEntity({ id: MAGS, description: 'Second rewrite.' }));
    await world.refreshLore().catch(() => 0);

    expect(await scoreOf(MAGS, fakeEmbedding('Mags Holloway: First rewrite.')) ?? 0).toBeLessThan(0.9);
  });

  it('makes added entities searchable by name and description', async () => {
    expectOk(await world.setEntity({ name: 'Copper Coin', type: 'ITEM', description: 'A dented coin.' }));
    await world.refreshLore();

    expect(await top(fakeEmbedding('Copper Coin'))).toBe('item:copper-coin');
    expect(await top(fakeEmbedding('Copper Coin: A dented coin.'))).toBe('item:copper-coin');
  });

  it('hides the story name vector of a renamed entity', async () => {
    const entity = (await world.state.entity(PIP))!;
    await db.transaction((tx) => putEntity(tx, 99, { ...entity, name: 'Pip the Retired' }));
    await world.refreshLore();

    expect(await scoreOf(PIP, await storyVector(PIP, 'name')) ?? 0).toBeLessThan(0.9);
    expect(await top(fakeEmbedding('Pip the Retired'))).toBe(PIP);
  });

  it('leaves the player out: their description is not lore', async () => {
    expectOk(await world.setEntity({ id: 'player', description: 'Aria, soaked to the bone.' }));
    expect(await world.refreshLore()).toBe(0);
    expect(embedded).toEqual([]);
  });

  it('catches up on vectors that were missing when the save is opened again', async () => {
    online = false;
    expectOk(await world.setEntity({ id: MAGS, description: NEW_MAGS }));
    await world.refreshLore().catch(() => 0);

    online = true;
    world = await openWorld(); // what Game.open does: open, then refresh in the background
    expect(await world.refreshLore()).toBe(1);
    expect(await top(fakeEmbedding(`Mags Holloway: ${NEW_MAGS}`))).toBe(MAGS);
  });
});
