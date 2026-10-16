import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { Db, SqlParams } from '@/core/storage/db';
import { openVectorIndex } from '@/core/vector/open-index';
import { toVector } from '@/core/vector/vector-index';
import type { Outcome } from '@/core/world/model';
import { World } from '@/core/world/world';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

const TAVERN = 'loc:rusty-gear-tavern';
const LAMPLIGHT = 'loc:lamplight-row';
const PLAYER = 'player';
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
  it('places the player at the start, leaving the story in its own tables', async () => {
    expect(await count('SELECT count(*) AS n FROM world_entities')).toBe(75); // 74 + player
    expect(await count('SELECT count(*) AS n FROM dyn_entities')).toBe(1); // only the player
    expect(await count('SELECT count(*) AS n FROM entities')).toBe(74);
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
    const event = expectOk(await world.setRelation('player', 'LOCATED_IN', 'Lamplight Row'));
    expect(event).toMatchObject({ kind: 'player.moved', payload: { from: TAVERN, to: LAMPLIGHT } });
    expect((await world.scene()).location?.id).toBe(LAMPLIGHT);
    expect(
      await count("SELECT updated_seq AS n FROM dyn_relations WHERE src = ? AND type = 'LOCATED_IN'", [
        PLAYER,
      ]),
    ).toBe(event.seq);

    expectOk(await world.setRelation('player', 'LOCATED_IN', TAVERN)); // the reverse edge
    expect((await world.scene()).location?.id).toBe(TAVERN);
  });

  it('refuses a location that is not adjacent, listing the exits', async () => {
    const outcome = await world.setRelation('player', 'LOCATED_IN', 'loc:orrery-vault');
    expectError(outcome, /no path from The Rusty Gear Tavern.*Exits: Lamplight Row/);
    expect((await world.scene()).location?.id).toBe(TAVERN);
    expect(await world.log.count('player.moved')).toBe(0);
  });

  it('understands partial and misspelled names', async () => {
    expectOk(await world.setRelation('player', 'LOCATED_IN', 'lamplight'));
    expectOk(await world.setRelation('player', 'LOCATED_IN', 'the rusty gear taverm'));
    expect((await world.scene()).location?.id).toBe(TAVERN);
  });
});

describe('enclosed places (ADR 31)', () => {
  it('lets the player walk into a place inside a place and meet who is there', async () => {
    expectOk(await world.setRelation('player', 'LOCATED_IN', LAMPLIGHT));
    expectOk(await world.setRelation('player', 'LOCATED_IN', 'Brassworks Alley'));
    const alley = await world.scene();
    expect(alley.exits.find((e) => e.to.id === 'loc:fenns-clockworks')?.description).toBe(
      "into Fenn's Clockworks",
    );

    expectOk(await world.setRelation('player', 'LOCATED_IN', "Fenn's Clockworks"));
    expect(ids((await world.scene()).here)).toContain('npc:tobias-fenn');
    expectOk(await world.setRelation('player', 'LOCATED_IN', 'loc:brassworks-alley'));
  });
});

describe('item.taken', () => {
  it('moves an item from the location to the inventory', async () => {
    const event = expectOk(await world.setRelation('player', 'HAS_ITEM', 'the brass lantern'));
    expect(event).toMatchObject({ payload: { item: 'item:brass-lantern', from: TAVERN } });
    const scene = await world.scene();
    expect(ids(scene.inventory)).toEqual(['item:brass-lantern']);
    expect(ids(scene.here)).not.toContain('item:brass-lantern');
  });

  it('refuses an item that lies in another location', async () => {
    expectError(await world.setRelation('player', 'HAS_ITEM', 'pneumatic post'), /no "pneumatic post" here/);
    expectError(await world.setRelation('player', 'HAS_ITEM', 'item:pneumatic-post'), /here/);
    expect(await world.log.count('item.taken')).toBe(0);
  });

  it('refuses fixed things, people and what the player already has', async () => {
    expectError(await world.setRelation('player', 'HAS_ITEM', 'crate'), /fixed in place/);
    expectError(await world.setRelation('player', 'HAS_ITEM', 'Mags Holloway'), /cannot be carried/);
    expectOk(await world.setRelation('player', 'HAS_ITEM', 'brass lantern'));
    expectError(await world.setRelation('player', 'HAS_ITEM', 'brass lantern'), /already has brass lantern/i);
  });

  it('takes an item out of a container that is here', async () => {
    await db.run(
      // a story where the canister lies in the crate (the story's own graph, before the game reads it)
      "UPDATE relations SET dst = 'item:crate' WHERE src = 'item:smoke-canister' AND type = 'LOCATED_IN'",
    );
    const event = expectOk(await world.setRelation('player', 'HAS_ITEM', 'smoke canister'));
    expect(event.payload).toEqual({ item: 'item:smoke-canister', from: 'item:crate' });
    expect(await world.state.placesOf('item:smoke-canister')).toEqual([PLAYER]);
  });

  it("takes an NPC's item only after that NPC reacted this turn (ADR 32, 34)", async () => {
    expectOk(await world.setRelation('player', 'LOCATED_IN', LAMPLIGHT));
    expectError(await world.setRelation('player', 'HAS_ITEM', 'calling card'), /Pip Quillon has calling card.*react_entity/);
    world.beginTurn();
    expectOk(
      await world.recordReaction({
        entity: 'npc:pip-quillon',
        situation: 'The stranger snatches the calling card from your fingers.',
        reaction: '"Oi! Fine, keep it — it opens no doors without me."',
      }),
    );
    const event = expectOk(await world.setRelation('player', 'HAS_ITEM', 'calling card'));
    expect(event.payload).toEqual({ item: 'doc:calling-card', from: 'npc:pip-quillon' });
    expect(ids(await world.state.inventory('npc:pip-quillon'))).toEqual([]);
    expect(ids(await world.state.inventory(PLAYER))).toEqual(['doc:calling-card']);
  });

  it("does not count another NPC's or an earlier turn's reaction", async () => {
    expectOk(await world.setRelation('player', 'LOCATED_IN', LAMPLIGHT));
    expectOk(await world.recordReaction({ entity: 'npc:pip-quillon', situation: 's', reaction: 'r' }));
    world.beginTurn();
    expectError(await world.setRelation('player', 'HAS_ITEM', 'calling card'), /Play out the attempt first/);
    expectOk(await world.recordReaction({ entity: 'npc:mags-holloway', situation: 's', reaction: 'r' }));
    expectError(await world.setRelation('player', 'HAS_ITEM', 'calling card'), /Play out the attempt first/);
    expect(await world.log.count('item.taken')).toBe(0);
  });
});

describe('item.dropped and item.moved', () => {
  it('puts an item from the inventory down here or into something here', async () => {
    expectOk(await world.setRelation('player', 'HAS_ITEM', 'brass lantern'));
    expectOk(await world.setRelation('player', 'LOCATED_IN', LAMPLIGHT));
    const event = expectOk(await world.setRelation('lantern', 'LOCATED_IN', LAMPLIGHT));
    expect(event).toMatchObject({ kind: 'item.dropped', payload: { item: 'item:brass-lantern', to: LAMPLIGHT } });
    const scene = await world.scene();
    expect(scene.inventory).toEqual([]);
    expect(ids(scene.here)).toContain('item:brass-lantern');
  });

  it('refuses to put down what the player carries anywhere but here', async () => {
    expectOk(await world.setRelation('player', 'HAS_ITEM', 'brass lantern'));
    expectError(
      await world.setRelation('brass lantern', 'LOCATED_IN', LAMPLIGHT),
      /put brass lantern down only here \(The Rusty Gear Tavern\)/,
    );
    expect(ids(await world.state.inventory(PLAYER))).toEqual(['item:brass-lantern']);
  });

  it('lets the master move things that are not in the player\'s hands, and hand them over', async () => {
    const moved = expectOk(await world.setRelation('brass lantern', 'LOCATED_IN', LAMPLIGHT));
    expect(moved).toMatchObject({ kind: 'item.moved', payload: { item: 'item:brass-lantern', from: TAVERN, to: LAMPLIGHT } });
    const given = expectOk(await world.setRelation('npc:mags-holloway', 'HAS_ITEM', 'item:brass-lantern'));
    expect(given).toMatchObject({ kind: 'item.moved', payload: { from: LAMPLIGHT, to: 'npc:mags-holloway' } });
    expect(ids(await world.state.inventory('npc:mags-holloway'))).toContain('item:brass-lantern');
  });

  it('gives what the player carries only to someone here', async () => {
    expectOk(await world.setRelation('player', 'HAS_ITEM', 'brass lantern'));
    expectError(await world.setRelation('npc:pip-quillon', 'HAS_ITEM', 'brass lantern'), /Pip Quillon is not here/);
    expectOk(await world.setRelation('npc:mags-holloway', 'HAS_ITEM', 'brass lantern'));
    expect(await world.state.inventory(PLAYER)).toEqual([]);
  });

  it('refuses places, paths and unknown relations', async () => {
    expectError(await world.setRelation(LAMPLIGHT, 'LOCATED_IN', TAVERN), /places stay where the story put them/);
    expectError(await world.setRelation(TAVERN, 'LEADS_TO', 'loc:orrery-vault'), /Paths between places come from the story/);
    expectError(await world.setRelation('player', 'OWNS', 'brass lantern'), /Unknown relation OWNS/);
    expectError(await world.setRelation('brass lantern', 'LOCATED_IN', 'npc:mags-holloway'), /use HAS_ITEM/);
  });
});

describe('entity.added', () => {
  it('creates a new thing in the current location; set_relation puts it in hand', async () => {
    const event = expectOk(
      await world.setEntity({ name: 'Spilled Ale', type: 'OTHER', description: 'A sticky puddle.' }),
    );
    expect(event.payload).toEqual({
      entity: 'ent:spilled-ale',
      relation: { type: 'LOCATED_IN', target: TAVERN },
    });
    expect(ids((await world.scene()).here)).toContain('ent:spilled-ale');

    expectOk(await world.setEntity({ name: 'Copper Coin', type: 'item', description: '' }));
    expectOk(await world.setRelation('player', 'HAS_ITEM', 'copper coin'));
    expect(ids((await world.scene()).inventory)).toEqual(['item:copper-coin']);
  });

  it('refuses duplicates, new locations and creating without all fields', async () => {
    expectError(
      await world.setEntity({ name: 'the Brass Lantern', type: 'ITEM', description: '' }),
      /already exists \(item:brass-lantern\)/,
    );
    expectError(await world.setEntity({ name: 'Secret Room', type: 'LOCATION', description: '' }), /Cannot create a LOCATION/);
    expectError(await world.setEntity({ name: 'Tiny Man', type: 'npc' }), /give its name, type and description/);
    expectError(await world.setEntity({ id: 'item:nothing-like-it', description: 'x' }), /leave out id/);
  });
});

describe('entity.edited', () => {
  it('changes the fields given, keeping before/after', async () => {
    const event = expectOk(await world.setEntity({ id: 'brass lantern', description: 'Dented now.' }));
    expect(event.payload).toEqual({
      id: 'item:brass-lantern',
      before: { description: expect.any(String) },
      after: { description: 'Dented now.' },
    });
    expect(await world.state.entity('item:brass-lantern')).toMatchObject({ description: 'Dented now.' });
    const renamed = expectOk(await world.setEntity({ id: 'item:brass-lantern', name: 'Dented Lantern' }));
    expect(renamed.payload).toMatchObject({ after: { name: 'Dented Lantern' } });
  });

  it('keeps the type, refuses taken names, and needs a change', async () => {
    expectError(await world.setEntity({ id: 'brass lantern', type: 'npc' }), /type cannot change/);
    expectError(await world.setEntity({ id: 'brass lantern', name: 'Mags Holloway' }), /already exists/);
    expectError(await world.setEntity({ id: 'brass lantern' }), /Nothing to change/);
  });
});

describe('the player as the entity "player" (ADR 43)', () => {
  it('is changed like any entity', async () => {
    const event = expectOk(await world.setEntity({ id: 'player', description: 'Aria, soaked to the bone.' }));
    expect(event).toMatchObject({
      kind: 'entity.edited',
      payload: { id: PLAYER, before: { description: 'A wandering cartographer.' }, after: { description: 'Aria, soaked to the bone.' } },
    });
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

describe('npc.moved', () => {
  it('sends a person to any location, out of sight too; the story keeps its own placement', async () => {
    const event = expectOk(await world.setRelation('Mags Holloway', 'LOCATED_IN', 'Orrery Vault'));
    expect(event).toMatchObject({
      kind: 'npc.moved',
      payload: { npc: 'npc:mags-holloway', from: TAVERN, to: 'loc:orrery-vault' },
    });
    expect(ids((await world.scene()).here)).not.toContain('npc:mags-holloway');
    expect((await world.state.locationOf('npc:mags-holloway'))?.id).toBe('loc:orrery-vault');
    const storyPlacement = "SELECT count(*) AS n FROM relations WHERE src = ? AND dst = ? AND type = 'LOCATED_IN'";
    expect(await count(storyPlacement, ['npc:mags-holloway', TAVERN])).toBe(1);
  });

  it('refuses places that are not locations, and staying put', async () => {
    expectError(
      await world.setRelation('npc:mags-holloway', 'LOCATED_IN', 'brass lantern'),
      /can only go to a location; brass lantern is not one/,
    );
    expectError(await world.setRelation('npc:mags-holloway', 'LOCATED_IN', TAVERN), /already in The Rusty Gear Tavern/);
    expect(await world.log.count('npc.moved')).toBe(0);
  });
});

describe('relation.set and relation.cancelled', () => {
  const MAGS = 'npc:mags-holloway';
  const PIP = 'npc:pip-quillon';
  const tieOf = async (id: string, other: string) =>
    (await world.state.ties(id)).find((t) => t.other.id === other)?.description;

  it('records a new tie, with the player too', async () => {
    const event = expectOk(await world.setRelation(MAGS, 'INTERACTS_WITH', 'Aria', 'refused to join the heist; not to be trusted'));
    expect(event.payload).toEqual({
      src: MAGS,
      dst: PLAYER,
      type: 'INTERACTS_WITH',
      description: 'refused to join the heist; not to be trusted',
      before: null,
    });
    expect(await tieOf(MAGS, PLAYER)).toBe('refused to join the heist; not to be trusted');
  });

  it("rewrites the story's tie between the two, whichever way it was written", async () => {
    const event = expectOk(await world.setRelation(PIP, 'INTERACTS_WITH', MAGS, 'quit her crew after the stranger walked away'));
    expect(event.payload).toMatchObject({ src: MAGS, dst: PIP, before: 'leader of the crew' });
    expect(await tieOf(PIP, MAGS)).toBe('quit her crew after the stranger walked away');
    expect((await world.state.ties(MAGS)).filter((t) => t.other.id === PIP)).toHaveLength(1);
  });

  it('cancels the ties between two, and refuses when there are none', async () => {
    const event = expectOk(await world.cancelRelation('Pip Quillon', 'Mags Holloway'));
    expect(event.payload).toEqual({
      a: PIP,
      b: MAGS,
      removed: [{ src: MAGS, dst: PIP, type: 'INTERACTS_WITH', description: 'leader of the crew' }],
    });
    expect(await tieOf(MAGS, PIP)).toBeUndefined();
    expectError(await world.cancelRelation(PIP, MAGS), /have no tie/);
  });

  it('refuses an empty tie and a tie with oneself', async () => {
    expectError(await world.setRelation(MAGS, 'INTERACTS_WITH', PIP, '  '), /Describe the tie/);
    expectError(await world.setRelation(MAGS, 'INTERACTS_WITH', 'Mags Holloway', 'self-respect'), /two different parties/);
    expectError(await world.setRelation(MAGS, 'INTERACTS_WITH', 'nobody at all', 'x'), /no "nobody at all"/);
  });
});

describe('notes and entity.reacted', () => {
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

  it('keeps actor notes apart from the master\'s: not in his notebook, not closable by him', async () => {
    expectOk(await world.addNote('Meet Mags at midnight.'));
    const noted = expectOk(await world.addActorNote('npc:mags-holloway', 'I will not trust the stranger again.'));
    expect(noted.payload).toEqual({ id: 2, text: 'I will not trust the stranger again.', entity: 'npc:mags-holloway' });
    expectOk(await world.addActorNote('npc:pip-quillon', 'Owes me a drink.'));
    expect((await world.log.openNotes()).map((n) => n.id)).toEqual([1]);
    expect((await world.log.actorNotes('npc:mags-holloway', 10)).map((n) => n.text)).toEqual([
      'I will not trust the stranger again.',
    ]);
    expectError(await world.closeNote(2), /no note #2/);
    expectError(await world.addActorNote('npc:nobody', 'x'), /no npc:nobody/);
  });

  it('journals an NPC reaction without touching the state', async () => {
    const before = await count('SELECT sum(updated_seq) AS n FROM dyn_entities');
    const event = expectOk(
      await world.recordReaction({
        entity: 'npc:mags-holloway',
        situation: 'The player snatches the plan off her table.',
        reaction: '"Hands off, rookie." She pins your wrist to the table.',
      }),
    );
    expect(event.kind).toBe('entity.reacted');
    expect(await count('SELECT sum(updated_seq) AS n FROM dyn_entities')).toBe(before);
    expectError(
      await world.recordReaction({ entity: 'npc:nobody', situation: '?', reaction: '' }),
      /no npc:nobody/,
    );
  });
});

describe('journal', () => {
  it('keeps the last 5 events in order with turn numbers, and resumes the turn counter', async () => {
    world.beginTurn();
    expectOk(await world.setRelation('player', 'HAS_ITEM', 'brass lantern'));
    expectOk(await world.setRelation('player', 'LOCATED_IN', LAMPLIGHT));
    world.beginTurn();
    expectOk(await world.setRelation('brass lantern', 'LOCATED_IN', LAMPLIGHT));
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
    const failing = failOn(db, 'INSERT INTO dyn_relations');
    const broken = await World.open(failing);
    await expect(broken.setRelation('player', 'LOCATED_IN', LAMPLIGHT)).rejects.toThrow('disk on fire');
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
