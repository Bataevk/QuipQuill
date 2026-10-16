import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { Db } from '@/core/storage/db';
import { relate, unrelate } from '@/core/world/dynamic-graph';
import type { Outcome } from '@/core/world/model';
import { World } from '@/core/world/world';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

const TAVERN = 'loc:rusty-gear-tavern';
const LAMPLIGHT = 'loc:lamplight-row';
const MAGS = 'npc:mags-holloway';
const PIP = 'npc:pip-quillon';
const CLOCKWORKS = 'loc:fenns-clockworks';
const ALLEY = 'loc:brassworks-alley';
const CHARACTER = { id: 'c1', name: 'Aria', description: 'A wandering cartographer.' };

let env: TestEnv;
let db: Db;
let world: World;

beforeEach(async () => {
  env = await makeTestEnv();
  await env.library.importer.import(DEMO_STORY);
  const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 't' });
  db = await env.library.saves.open(save.id);
  world = await World.open(db);
  world.beginTurn();
  expectOk(await world.startSession({ storyId: DEMO_STORY_ID, character: CHARACTER, startLocation: TAVERN }));
});

afterEach(async () => {
  await db?.close();
  await env?.cleanup();
});

function expectOk<T>(outcome: Outcome<T>): T {
  if (!outcome.ok) throw new Error(`expected ok, got: ${outcome.error}`);
  return outcome.value;
}

/** Writes straight to the dynamic graph, as a command would (seq 99). */
const write = (fn: (tx: Db) => Promise<void>) => db.transaction(fn);
const between = (src: string, dst: string) =>
  db.all<{ id: number; type: string; description: string }>(
    'SELECT id, type, description FROM world_relations WHERE src = ? AND dst = ? ORDER BY type',
    [src, dst],
  );
const count = async (sql: string, params: string[] = []) =>
  Number((await db.get<{ n: number }>(sql, params))?.n);
const storyBetween = (src: string, dst: string) =>
  db.all<{ type: string }>('SELECT type FROM relations WHERE src = ? AND dst = ? ORDER BY type', [src, dst]);

/** The story's own graph, row by row (BLOB vectors included). */
async function storyGraph(of: Db) {
  return {
    entities: await of.all('SELECT * FROM entities ORDER BY id'),
    relations: await of.all('SELECT * FROM relations ORDER BY id'),
    embeddings: await of.all('SELECT * FROM embeddings ORDER BY entity_id, kind'),
  };
}

describe('the static graph', () => {
  it('stays exactly as the story file has it after a playthrough', async () => {
    expectOk(await world.setRelation('player', 'HAS_ITEM', 'brass lantern'));
    expectOk(await world.setEntity({ id: MAGS, description: 'Wary of the stranger now.' }));
    expectOk(await world.setEntity({ name: 'Spilled Ale', type: 'OTHER', description: 'A sticky puddle.' }));
    expectOk(await world.setEntity({ id: 'player', description: 'Aria, soaked.' }));
    expectOk(await world.setRelation('player', 'LOCATED_IN', LAMPLIGHT));
    expectOk(await world.setRelation('brass lantern', 'LOCATED_IN', LAMPLIGHT));
    await write(async (tx) => {
      await unrelate(tx, 99, MAGS, PIP, 'INTERACTS_WITH');
      await relate(tx, 99, MAGS, 'npc:silas-ashcombe', 'INTERACTS_WITH', 'made peace with Silas');
    });

    const original = await env.driver.open(DEMO_STORY, { readOnly: true });
    try {
      expect(await storyGraph(db)).toEqual(await storyGraph(original));
    } finally {
      await original.close();
    }
    expect((await world.state.entity(MAGS))?.description).toBe('Wary of the stranger now.');
    expect((await world.scene()).here.map((e) => e.id)).toContain('item:brass-lantern');
  });
});

describe('CANCEL', () => {
  it('hides exactly one story relation, the others between the pair stay', async () => {
    expect((await between(CLOCKWORKS, ALLEY)).map((r) => r.type)).toEqual(['LEADS_TO', 'LOCATED_IN']);
    await write((tx) => unrelate(tx, 99, CLOCKWORKS, ALLEY, 'LOCATED_IN'));

    expect((await between(CLOCKWORKS, ALLEY)).map((r) => r.type)).toEqual(['LEADS_TO']);
    expect(await storyBetween(CLOCKWORKS, ALLEY)).toEqual([{ type: 'LEADS_TO' }, { type: 'LOCATED_IN' }]);
    const story = await db.get<{ id: number }>(
      "SELECT id FROM relations WHERE src = ? AND dst = ? AND type = 'LOCATED_IN'",
      [CLOCKWORKS, ALLEY],
    );
    expect(await db.all('SELECT id, type, description FROM dyn_relations WHERE src = ?', [CLOCKWORKS])).toEqual([
      { id: story!.id, type: 'CANCEL', description: 'LOCATED_IN' },
    ]);
  });

  it('cancels two relations of one pair separately', async () => {
    await write(async (tx) => {
      await unrelate(tx, 99, CLOCKWORKS, ALLEY, 'LOCATED_IN');
      await unrelate(tx, 99, CLOCKWORKS, ALLEY, 'LEADS_TO');
      await unrelate(tx, 99, CLOCKWORKS, ALLEY, 'LEADS_TO'); // twice is still one
    });
    expect(await between(CLOCKWORKS, ALLEY)).toEqual([]);
    expect(await count("SELECT count(*) AS n FROM dyn_relations WHERE type = 'CANCEL'")).toBe(2);
  });

  it('is lifted when the relation is written again: the story relation comes back as it was', async () => {
    const [before] = await between('item:brass-lantern', TAVERN);
    expectOk(await world.setRelation('player', 'HAS_ITEM', 'brass lantern'));
    expect(await between('item:brass-lantern', TAVERN)).toEqual([]);
    expectOk(await world.setRelation('brass lantern', 'LOCATED_IN', TAVERN));

    expect(await between('item:brass-lantern', TAVERN)).toEqual([before]);
    expect(await db.all("SELECT * FROM dyn_relations WHERE src = 'item:brass-lantern'")).toEqual([]);
  });

  it('is not needed for a dynamic relation: that one is simply removed', async () => {
    expectOk(await world.setEntity({ name: 'Copper Coin', type: 'ITEM', description: '' }));
    expectOk(await world.setRelation('player', 'HAS_ITEM', 'copper coin'));
    expect(await db.all("SELECT type FROM dyn_relations WHERE src = 'item:copper-coin'")).toEqual([]);
  });
});

describe('the dynamic graph', () => {
  it('overrides a story entity as a whole, the story row stays', async () => {
    expectOk(await world.setEntity({ id: MAGS, description: 'Done with conspiracies.' }));
    expect(await world.state.entity(MAGS)).toMatchObject({
      name: 'Mags Holloway',
      type: 'NPC',
      description: 'Done with conspiracies.',
    });
    expect(await count('SELECT count(*) AS n FROM world_entities WHERE id = ?', [MAGS])).toBe(1);
    const story = await db.get<{ description: string }>('SELECT description FROM entities WHERE id = ?', [MAGS]);
    expect(story?.description).not.toBe('Done with conspiracies.');
  });

  it("overrides a story relation's description in place, keeping the order of ties", async () => {
    const order = (await world.state.ties(MAGS)).map((t) => t.other.id);
    await write((tx) => relate(tx, 99, MAGS, PIP, 'INTERACTS_WITH', 'once led his crew; parted ways'));

    const ties = await world.state.ties(MAGS);
    expect(ties.map((t) => t.other.id)).toEqual(order);
    expect(ties.find((t) => t.other.id === PIP)?.description).toBe('once led his crew; parted ways');
    expect(await between(MAGS, PIP)).toHaveLength(1);
  });

  it('adds new ties after the story ones and drops cancelled ones', async () => {
    const order = (await world.state.ties(MAGS)).map((t) => t.other.id);
    await write(async (tx) => {
      await unrelate(tx, 99, MAGS, PIP, 'INTERACTS_WITH');
      await relate(tx, 99, MAGS, 'player', 'INTERACTS_WITH', 'turned down her offer to join the crew');
    });
    const ties = await world.state.ties(MAGS);
    expect(ties.map((t) => t.other.id)).toEqual([...order.filter((id) => id !== PIP), 'player']);
  });
});
