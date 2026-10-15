import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { Db, SqlParams } from '@/core/storage/db';
import { openVectorIndex } from '@/core/vector/open-index';
import { toVector } from '@/core/vector/vector-index';
import type { Outcome } from '@/core/world/model';
import { World } from '@/core/world/world';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

const TAVERN = 'loc:rusty-gear-tavern';
const LAMPLIGHT = 'loc:lamplight-row';
const PLAYER = 'player:c1';
const CHARACTER = {
  id: 'c1',
  name: 'Aria',
  sex: 'female',
  age: 27,
  race: 'elf',
  description: 'A wandering cartographer.',
};

let env: TestEnv;
let db: Db;
let world: World;

beforeEach(async () => {
  env = await makeTestEnv();
  await env.library.importer.import(DEMO_STORY);
  await env.library.repository.addCharacter({ ...CHARACTER, createdAt: '2026-10-15T00:00:00.000Z' });
  const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: 'c1', title: 't' });
  db = await env.library.saves.open(save.id);
  world = await World.open(db, { vectors: await openVectorIndex(db) });
  world.beginTurn();
  expectOk(
    await world.startSession({ storyId: DEMO_STORY_ID, character: CHARACTER, startLocation: TAVERN }),
  );
});

afterEach(async () => {
  await db?.close();
  await env?.cleanup();
});

function expectOk<T>(outcome: Outcome<T>): T {
  if (!outcome.ok) throw new Error(`expected ok, got: ${outcome.error}`);
  return outcome.value;
}

function expectError(outcome: Outcome<unknown>, pattern: RegExp): void {
  expect(outcome.ok).toBe(false);
  if (!outcome.ok) expect(outcome.error).toMatch(pattern);
}

const ids = (entities: { id: string }[]) => entities.map((e) => e.id);
const count = async (sql: string, params: SqlParams = []) =>
  Number((await db.get<{ n: number }>(sql, params))?.n);

describe('session.started', () => {
  it('copies the story into state and places the player at the start', async () => {
    expect(await count('SELECT count(*) AS n FROM state_entities')).toBe(75); // 74 + player
    const scene = await world.scene();
    expect(scene.player).toMatchObject({ id: PLAYER, type: 'PLAYER', name: 'Aria' });
    expect(scene.player.props).toMatchObject({ race: 'elf', original: CHARACTER });
    expect(scene.location?.id).toBe(TAVERN);
    expect(ids(scene.exits.map((e) => e.to))).toEqual([LAMPLIGHT]);
    expect(ids(scene.here)).toEqual(
      expect.arrayContaining(['npc:mags-holloway', 'item:brass-lantern', 'item:crate']),
    );
    expect(ids(scene.here)).not.toContain(PLAYER);
    expect(scene.recent.map((e) => e.kind)).toEqual(['session.started']);
  });

  it('happens once', async () => {
    expectError(
      await world.startSession({ storyId: DEMO_STORY_ID, character: CHARACTER, startLocation: TAVERN }),
      /already started/,
    );
    expect(await world.log.count()).toBe(1);
  });
});

describe('player.moved', () => {
  it('follows a LEADS_TO path, by name or id', async () => {
    const event = expectOk(await world.move('Lamplight Row'));
    expect(event).toMatchObject({ kind: 'player.moved', payload: { from: TAVERN, to: LAMPLIGHT } });
    expect((await world.scene()).location?.id).toBe(LAMPLIGHT);
    expect(
      await count("SELECT updated_seq AS n FROM state_relations WHERE src = ? AND type = 'LOCATED_IN'", [
        PLAYER,
      ]),
    ).toBe(event.seq);

    expectOk(await world.move(TAVERN)); // the reverse edge
    expect((await world.scene()).location?.id).toBe(TAVERN);
  });

  it('refuses a location that is not adjacent, listing the exits', async () => {
    const outcome = await world.move('loc:orrery-vault');
    expectError(outcome, /no path from The Rusty Gear Tavern.*Exits: Lamplight Row/);
    expect((await world.scene()).location?.id).toBe(TAVERN);
    expect(await world.log.count('player.moved')).toBe(0);
  });

  it('understands partial and misspelled names', async () => {
    expectOk(await world.move('lamplight'));
    expectOk(await world.move('the rusty gear taverm'));
    expect((await world.scene()).location?.id).toBe(TAVERN);
  });
});

describe('enclosed places (ADR 31)', () => {
  it('lets the player walk into a place inside a place and meet who is there', async () => {
    expectOk(await world.move(LAMPLIGHT));
    expectOk(await world.move('Brassworks Alley'));
    const alley = await world.scene();
    expect(alley.exits.find((e) => e.to.id === 'loc:fenns-clockworks')?.description).toBe(
      "into Fenn's Clockworks",
    );

    expectOk(await world.move("Fenn's Clockworks"));
    expect(ids((await world.scene()).here)).toContain('npc:tobias-fenn');
    expectOk(await world.move('loc:brassworks-alley'));
  });
});

describe('item.taken', () => {
  it('moves an item from the location to the inventory', async () => {
    const event = expectOk(await world.take('the brass lantern'));
    expect(event).toMatchObject({ payload: { item: 'item:brass-lantern', from: TAVERN } });
    const scene = await world.scene();
    expect(ids(scene.inventory)).toEqual(['item:brass-lantern']);
    expect(ids(scene.here)).not.toContain('item:brass-lantern');
  });

  it('refuses an item that lies in another location', async () => {
    expectError(await world.take('pneumatic post'), /no "pneumatic post" here/);
    expectError(await world.take('item:pneumatic-post'), /here/);
    expect(await world.log.count('item.taken')).toBe(0);
  });

  it('refuses fixed things, people and what the player already has', async () => {
    expectError(await world.take('crate'), /fixed in place/);
    expectError(await world.take('Mags Holloway'), /cannot be carried/);
    expectOk(await world.take('brass lantern'));
    expectError(await world.take('brass lantern'), /already has brass lantern/i);
  });

  it('takes an item out of a container that is here', async () => {
    await db.run(
      "UPDATE state_relations SET dst = 'item:crate' WHERE src = 'item:smoke-canister' AND type = 'LOCATED_IN'",
    );
    const event = expectOk(await world.take('smoke canister'));
    expect(event.payload).toEqual({ item: 'item:smoke-canister', from: 'item:crate' });
    expect(await world.state.placesOf('item:smoke-canister')).toEqual([PLAYER]);
  });

  it("takes an NPC's item only after that NPC reacted this turn (ADR 32, 34)", async () => {
    expectOk(await world.move(LAMPLIGHT));
    expectError(await world.take('calling card'), /Pip Quillon has calling card.*npc_react/);
    world.beginTurn();
    expectOk(
      await world.recordNpcReaction({
        npc: 'npc:pip-quillon',
        situation: 'The stranger snatches the calling card from your fingers.',
        reaction: '"Oi! Fine, keep it — it opens no doors without me."',
      }),
    );
    const event = expectOk(await world.take('calling card'));
    expect(event.payload).toEqual({ item: 'doc:calling-card', from: 'npc:pip-quillon' });
    expect(ids(await world.state.inventory('npc:pip-quillon'))).toEqual([]);
    expect(ids(await world.state.inventory(PLAYER))).toEqual(['doc:calling-card']);
  });

  it("does not count another NPC's or an earlier turn's reaction", async () => {
    expectOk(await world.move(LAMPLIGHT));
    expectOk(await world.recordNpcReaction({ npc: 'npc:pip-quillon', situation: 's', reaction: 'r' }));
    world.beginTurn();
    expectError(await world.take('calling card'), /Play out the attempt first/);
    expectOk(await world.recordNpcReaction({ npc: 'npc:mags-holloway', situation: 's', reaction: 'r' }));
    expectError(await world.take('calling card'), /Play out the attempt first/);
    expect(await world.log.count('item.taken')).toBe(0);
  });
});

describe('item.dropped', () => {
  it('puts an item from the inventory into the current location', async () => {
    expectOk(await world.take('brass lantern'));
    expectOk(await world.move(LAMPLIGHT));
    const event = expectOk(await world.drop('lantern'));
    expect(event.payload).toEqual({ item: 'item:brass-lantern', to: LAMPLIGHT });
    const scene = await world.scene();
    expect(scene.inventory).toEqual([]);
    expect(ids(scene.here)).toContain('item:brass-lantern');
  });

  it('refuses what the player does not have', async () => {
    expectError(await world.drop('brass lantern'), /does not have "brass lantern". Inventory: empty/);
  });
});

describe('entity.added', () => {
  it('adds a new thing to the scene or the inventory', async () => {
    const event = expectOk(
      await world.addEntity({ name: 'Spilled Ale', type: 'OTHER', description: 'A sticky puddle.' }),
    );
    expect(event.payload).toEqual({
      entity: 'ent:spilled-ale',
      relation: { type: 'LOCATED_IN', target: TAVERN },
    });
    expect(ids((await world.scene()).here)).toContain('ent:spilled-ale');

    expectOk(
      await world.addEntity({ name: 'Copper Coin', type: 'ITEM', description: '', inInventory: true }),
    );
    expect(ids((await world.scene()).inventory)).toEqual(['item:copper-coin']);
  });

  it('refuses duplicates, new locations and people in pockets', async () => {
    expectError(
      await world.addEntity({ name: 'the Brass Lantern', type: 'ITEM', description: '' }),
      /already exists \(item:brass-lantern\)/,
    );
    expectError(await world.addEntity({ name: 'Secret Room', type: 'LOCATION', description: '' }), /Cannot add a LOCATION/);
    expectError(
      await world.addEntity({ name: 'Tiny Man', type: 'NPC', description: '', inInventory: true }),
      /cannot be put in the inventory/,
    );
  });
});

describe('entity.edited', () => {
  it('changes the description and merges props, keeping before/after', async () => {
    const event = expectOk(
      await world.editEntity('brass lantern', { description: 'Dented now.', props: { lit: true } }),
    );
    expect(event.payload).toMatchObject({
      id: 'item:brass-lantern',
      before: { props: {} },
      after: { description: 'Dented now.', props: { lit: true } },
    });
    expect(await world.state.entity('item:brass-lantern')).toMatchObject({
      description: 'Dented now.',
      props: { lit: true },
    });
  });

  it('leaves the player to update_player and needs a change', async () => {
    expectError(await world.editEntity(PLAYER, { description: 'x' }), /update_player/);
    expectError(await world.editEntity('brass lantern', {}), /Nothing to change/);
  });
});

describe('player.updated and character.adapted', () => {
  it('updates the player description', async () => {
    const event = expectOk(await world.updatePlayer('Aria, soaked to the bone.'));
    expect(event.payload).toEqual({ before: 'A wandering cartographer.', after: 'Aria, soaked to the bone.' });
    expect((await world.state.player())?.description).toBe('Aria, soaked to the bone.');
  });

  it('adapts the character and keeps the original sheet', async () => {
    const event = expectOk(
      await world.adaptCharacter({
        adapted: { race: 'human', age: 31, description: 'A cartographer from the colonies.' },
        choice: 'adapted',
      }),
    );
    expect(event.payload).toMatchObject({ original: CHARACTER, choice: 'adapted' });
    expect(await world.state.player()).toMatchObject({
      name: 'Aria',
      description: 'A cartographer from the colonies.',
      props: { race: 'human', age: 31, sex: 'female', original: CHARACTER },
    });
  });

  it('adapts only once', async () => {
    expectOk(await world.adaptCharacter({ adapted: { name: 'Ari' }, choice: 'adapted' }));
    expectError(await world.adaptCharacter({ adapted: { name: 'Bea' }, choice: 'adapted' }), /already settled/);
    expect(await world.log.count('character.adapted')).toBe(1);
    expect((await world.state.player())?.name).toBe('Ari');
  });
});

describe('notes and npc.reacted', () => {
  it('adds and closes notes', async () => {
    const added = expectOk(await world.addNote('Meet Mags at midnight.'));
    expect(added.payload).toEqual({ id: 1, text: 'Meet Mags at midnight.' });
    expectOk(await world.addNote('Find the second key.'));
    expect((await world.log.openNotes()).map((n) => n.id)).toEqual([1, 2]);

    expectOk(await world.closeNote(1));
    expect((await world.log.openNotes()).map((n) => n.id)).toEqual([2]);
    expectError(await world.closeNote(1), /already closed/);
    expectError(await world.closeNote(9), /no note #9/);
  });

  it('journals an NPC reaction without touching the state', async () => {
    const before = await count('SELECT sum(updated_seq) AS n FROM state_entities');
    const event = expectOk(
      await world.recordNpcReaction({
        npc: 'npc:mags-holloway',
        situation: 'The player snatches the plan off her table.',
        reaction: '"Hands off, rookie." She pins your wrist to the table.',
      }),
    );
    expect(event.kind).toBe('npc.reacted');
    expect(await count('SELECT sum(updated_seq) AS n FROM state_entities')).toBe(before);
    expectError(
      await world.recordNpcReaction({ npc: 'npc:nobody', situation: '?', reaction: '' }),
      /no npc:nobody/,
    );
  });
});

describe('journal', () => {
  it('keeps the last 5 events in order with turn numbers, and resumes the turn counter', async () => {
    world.beginTurn();
    expectOk(await world.take('brass lantern'));
    expectOk(await world.move(LAMPLIGHT));
    world.beginTurn();
    expectOk(await world.drop('brass lantern'));
    expectOk(await world.addNote('n1'));
    expectOk(await world.addNote('n2'));

    const recent = await world.log.recent(5);
    expect(recent.map((e) => e.kind)).toEqual([
      'item.taken',
      'player.moved',
      'item.dropped',
      'note.added',
      'note.added',
    ]);
    expect(recent.map((e) => e.turn)).toEqual([2, 2, 3, 3, 3]);
    expect((await World.open(db)).currentTurn).toBe(3);
  });

  it('writes nothing when the state change fails mid-transaction', async () => {
    const failing = failOn(db, 'INSERT INTO state_relations');
    const broken = await World.open(failing);
    await expect(broken.move(LAMPLIGHT)).rejects.toThrow('disk on fire');
    expect(await world.log.count()).toBe(1); // only session.started
    expect((await world.scene()).location?.id).toBe(TAVERN);
  });
});

describe('lore search', () => {
  it('finds entities by meaning, one hit per entity', async () => {
    const row = await db.get<{ vector: Uint8Array }>(
      "SELECT vector FROM embeddings WHERE entity_id = 'npc:mags-holloway' AND kind = 'desc'",
    );
    const hits = expectOk(await world.searchLore(toVector(row!.vector), 3));
    expect(hits[0].entity.id).toBe('npc:mags-holloway');
    expect(new Set(ids(hits.map((h) => h.entity))).size).toBe(hits.length);
  });
});

/** A Db whose statements matching `fragment` throw — to prove commands are atomic. */
function failOn(inner: Db, fragment: string): Db {
  const guard = (sql: string) => {
    if (sql.includes(fragment)) throw new Error('disk on fire');
  };
  const wrap = (db: Db): Db => ({
    run: async (sql, params) => (guard(sql), db.run(sql, params)),
    all: (sql, params) => db.all(sql, params),
    get: (sql, params) => db.get(sql, params),
    exec: (script) => db.exec(script),
    transaction: (fn) => db.transaction((tx) => fn(wrap(tx))),
    close: () => db.close(),
  });
  return wrap(inner);
}
