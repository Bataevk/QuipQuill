/**
 * Stage L.5: real playthroughs of Alice's Adventures in Wonderland for the landing site — this
 * core, a local cloud and the real models through OpenRouter. Writes every transcript (with the
 * speakers' names and each move's world events) to .live-runs/alice-*.json; the site quotes them.
 * Costs real money (cents): `QQ_LIVE=1 npx vitest run src/core/game/alice.live.test.ts` after
 * `make infra-up`. QQ_RUNS sets how many runs as Alice (2 by default).
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import { afterAll, beforeAll, expect, it } from 'vitest';

import { AuthSession, MemoryTokenStore } from '@/core/net/auth';
import { CloudSocket } from '@/core/net/cloud-socket';
import { CloudEmbeddings } from '@/core/net/embeddings';
import { formatUsage } from '@/core/protocol/usage-meter';
import { newId, nowIso } from '@/core/util/ids';
import { startTestCloud, TEST_INVITE, type TestCloud } from '@/platform/node/cloud-server';
import { makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import type { TurnResult } from '../agent/run-turn';
import type { StoryMeta } from '../story/story-meta';
import { ALICE_MOVES, ALICE_PEOPLE_MOVES, OWN_HERO, OWN_HERO_MOVES } from './alice-script';
import { Game } from './game';

const ALICE_STORY = fileURLToPath(new URL('../../../../../stories/alice/alice.qqstory', import.meta.url));
const OUT_DIR = fileURLToPath(new URL('../../../.live-runs/', import.meta.url));
const RUNS = Number(process.env.QQ_RUNS ?? 2);
const LIVE = process.env.QQ_LIVE === '1';

let cloud: TestCloud;
let env: TestEnv;
let auth: AuthSession;
let socket: CloudSocket;
let storyId: string;
let heroes: StoryMeta['heroes'];

beforeAll(async () => {
  if (!LIVE) return;
  cloud = await startTestCloud({ live: true, dailyLimitUsd: 2 });
  env = await makeTestEnv();
  const imported = await env.library.importer.import(ALICE_STORY);
  storyId = imported.story.id;
  heroes = imported.meta.heroes;
  auth = new AuthSession(cloud.httpUrl, new MemoryTokenStore());
  await auth.register(`alice-${newId().slice(0, 8)}@quipquill-dev.app`, 'long enough pass', TEST_INVITE);
  socket = new CloudSocket({ url: cloud.wsUrl, auth });
  socket.start();
}, 90_000);

afterAll(async () => {
  socket?.stop();
  await env?.cleanup();
  await cloud?.stop();
});

async function openGame(characterId: string, title: string): Promise<Game> {
  const save = await env.library.saves.create({ storyId, characterId, title });
  return Game.open(
    { library: env.library, socket, embeddings: (model) => new CloudEmbeddings(cloud.httpUrl, auth, model) },
    save.id,
  );
}

interface Move {
  move: string;
  seconds: number;
  events: string[];
}

/** The transcript with names instead of channels, and what each move did to the world. */
async function save(file: string, game: Game, moves: Move[]): Promise<void> {
  const names = new Map<string, string>();
  const lines = [];
  for (const line of await game.lines()) {
    if (!names.has(line.speaker)) names.set(line.speaker, (await game.speakerOf(line.speaker)) ?? line.speaker);
    lines.push({ seq: line.seq, speaker: line.speaker, name: names.get(line.speaker), text: line.text });
  }
  await mkdir(OUT_DIR, { recursive: true });
  const data = { story: 'alice', hero: game.character?.name, cost: formatUsage(game.sessionUsage), moves, lines };
  await writeFile(`${OUT_DIR}${file}`, JSON.stringify(data, null, 2));
  console.log(`${file}: ${lines.length} lines, ${data.cost}`);
}

async function play(game: Game, moves: readonly string[], log: Move[]): Promise<void> {
  for (const move of moves) {
    const t0 = Date.now();
    const turn = (await game.send(move)) as TurnResult;
    await game.whenGenerated();
    const events = (await game.world.log.ofTurn(turn.turn)).map((e) => (e.actor === 'generator' ? `gen:${e.kind}` : e.kind));
    log.push({ move, seconds: Math.round((Date.now() - t0) / 100) / 10, events });
    // An empty reply is logged, not fatal: the rest of the run is still worth having.
    if (!turn.reply) console.log(`Empty reply to "${move}"`);
  }
}

it.skipIf(!LIVE)(`plays Alice ${RUNS} times as the story's own hero`, async () => {
  for (let run = 1; run <= RUNS; run++) {
    const [hero] = heroes;
    const character = await env.library.repository.upsertStoryHero(storyId, hero, nowIso());
    const game = await openGame(character.id, `Alice ${run}`);
    const log: Move[] = [];
    try {
      const opening = (await game.recover()) as TurnResult;
      expect(opening.reply).toBeTruthy();
      await play(game, process.env.QQ_SCRIPT === 'people' ? ALICE_PEOPLE_MOVES : ALICE_MOVES, log);
      await save(`alice-${process.env.QQ_RUN_TAG ?? ''}${run}.json`, game, log);
    } finally {
      await game.close();
    }
  }
});

it.skipIf(!LIVE || process.env.QQ_OWN_HERO === '0')('fits a hero of the player’s own into Wonderland', async () => {
  const characterId = newId();
  await env.library.repository.addCharacter({ id: characterId, ...OWN_HERO, createdAt: nowIso() });
  const game = await openGame(characterId, 'Max in Wonderland');
  const log: Move[] = [];
  try {
    await game.start();
    await game.choose(0);
    await play(game, OWN_HERO_MOVES, log);
    await save('alice-own-hero.json', game, log);
  } finally {
    await game.close();
  }
});
