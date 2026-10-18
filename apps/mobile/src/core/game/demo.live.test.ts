/**
 * Stage 6.1: the demo scenario end to end — this core, a local cloud, the real Haiku through
 * OpenRouter — three times in a row, with the cost of every move (the background generator's
 * included, stage 6.2d). Stage 6.2d: the player refuses the heist, and the world follows.
 * Costs real money (about $0.04 per run): `npm run test:live` after `make infra-up`
 * (`QQ_RUNS=0` skips the demo runs). One file, so the scenarios share the cloud one at a time.
 */
import { AIMessage, type BaseMessage } from '@langchain/core/messages';
import { afterAll, beforeAll, expect, it } from 'vitest';

import { AuthSession, MemoryTokenStore } from '@/core/net/auth';
import { CloudSocket } from '@/core/net/cloud-socket';
import { CloudEmbeddings } from '@/core/net/embeddings';
import { formatChannels, formatUsage } from '@/core/protocol/usage-meter';
import { newId, nowIso } from '@/core/util/ids';
import { startTestCloud, TEST_INVITE, type TestCloud } from '@/platform/node/cloud-server';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import { DEMO_CHARACTER, DEMO_MOVES } from './demo-script';
import { Game } from './game';
import type { TurnResult } from '../agent/run-turn';

const RUNS = Number(process.env.QQ_RUNS ?? 3);

let cloud: TestCloud;
let env: TestEnv;
let auth: AuthSession;
let socket: CloudSocket;

beforeAll(async () => {
  cloud = await startTestCloud({ live: true, dailyLimitUsd: 2 });
  env = await makeTestEnv();
  await env.library.importer.import(DEMO_STORY);
  auth = new AuthSession(cloud.httpUrl, new MemoryTokenStore());
  await auth.register(`demo-${newId().slice(0, 8)}@quipquill-dev.app`, 'long enough pass', TEST_INVITE);
  socket = new CloudSocket({ url: cloud.wsUrl, auth });
  socket.start();
}, 90_000);

afterAll(async () => {
  socket?.stop();
  await env?.cleanup();
  await cloud?.stop();
});

interface Step {
  step: string;
  ms: number;
  cost: number;
  usage: string;
  events: string[];
  reply: string;
}

async function openNewGame(title: string): Promise<Game> {
  const characterId = newId();
  await env.library.repository.addCharacter({ id: characterId, ...DEMO_CHARACTER, createdAt: nowIso() });
  const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId, title });
  return Game.open(
    { library: env.library, socket, embeddings: (model) => new CloudEmbeddings(cloud.httpUrl, auth, model) },
    save.id,
  );
}

/**
 * Runs an action and records it. `ms` is what the player waited; cost and events include the
 * background generator that follows a player's turn (`whenGenerated`), marked with `gen:`.
 */
function recorder(game: Game, steps: Step[]) {
  return async (step: string, action: () => Promise<{ reply?: string; events?: { kind: string }[] } | object>) => {
    const t0 = Date.now();
    const result = (await action()) as Partial<TurnResult> & { rationale?: string };
    const ms = Date.now() - t0;
    await game.whenGenerated();
    const events =
      result.turn === undefined
        ? (result.events ?? []).map((e) => e.kind)
        : (await game.world.log.ofTurn(result.turn)).map((e) => (e.actor === 'generator' ? `gen:${e.kind}` : e.kind));
    steps.push({
      step,
      ms,
      cost: game.lastUsage?.costUsd ?? 0,
      usage: game.lastUsage ? formatUsage(game.lastUsage) : '',
      events,
      reply: (result.reply ?? result.rationale ?? '').slice(0, 160).replace(/\s+/g, ' '),
    });
    return result;
  };
}

/** One message of a turn for diagnostics: type, tool calls, text. */
function describeMessage(m: BaseMessage): string {
  const calls = AIMessage.isInstance(m) ? (m.tool_calls ?? []).map((c) => `${c.name}(${JSON.stringify(c.args)})`) : [];
  const actor = m.additional_kwargs?.qq_actor ? ' [line]' : '';
  return `${m.getType()}${actor}: ${m.text.slice(0, 300)} ${calls.join(' ')}`.trim();
}

function table(title: string, steps: Step[]): string {
  return [`\n### ${title}`, '| Step | s | $ | Usage | Events | Reply |', '|---|---|---|---|---|---|']
    .concat(
      steps.map(
        (s) =>
          `| ${s.step} | ${(s.ms / 1000).toFixed(1)} | ${s.cost.toFixed(4)} | ${s.usage} | ${s.events.join(' ') || '—'} | ${s.reply.replace(/\|/g, '/')} |`,
      ),
    )
    .join('\n');
}

async function playOnce(run: number): Promise<Step[]> {
  const game = await openNewGame(`Demo ${run}`);
  const steps: Step[] = [];
  const timed = recorder(game, steps);
  try {
    const proposal = await timed('adaptation', () => game.start());
    expect((proposal as { options: unknown[] }).options.length).toBeGreaterThan(0);
    const opening = await timed('choice + opening scene', () => game.choose(0));
    expect(opening.reply).toBeTruthy();
    for (const move of DEMO_MOVES) {
      const turn = (await timed(move, () => game.send(move))) as TurnResult;
      if (!turn.reply) console.log(`Empty reply to "${move}":`, turn.messages.map(describeMessage));
      expect(turn.reply, move).toBeTruthy();
      expect(turn.cutShort, move).toBe(false);
    }
    expect(await game.nextStep()).toBeUndefined();
    const kinds = new Set(steps.flatMap((s) => s.events));
    console.log(`run ${run}: events ${[...kinds].join(', ')}; total ${formatUsage(game.sessionUsage)}`);
    console.log(`run ${run} by channel:\n  ${formatChannels(game.sessionUsage).join('\n  ')}`);
    return steps;
  } finally {
    await game.close();
  }
}

it(`plays the demo scenario ${RUNS} times in a row without errors`, async () => {
  for (let run = 1; run <= RUNS; run++) console.log(table(`Run ${run}`, await playOnce(run)));
});

const LUCK = [
  'I search under the tables and benches for anything useful someone dropped.',
  'I go back to the bar and look around the taproom.',
  'I ask the bartender for a mug of ale.',
] as const;

const REFUSAL = [
  'I ask Mags what exactly she needs from me tonight.',
  "I tell Mags I'm out: I won't rob a bank for anyone. She should find another locksmith.",
  'I ask Mags if she will at least buy me a drink for my trouble.',
  'I go up the stairs to Lamplight Row.',
  'I ask Pip whether Mags has told him about me.',
] as const;

it('stages 6.2d, 6.2h: after the player refuses the heist, the world and the people follow', async () => {
  const game = await openNewGame('Refusal');
  const steps: Step[] = [];
  const timed = recorder(game, steps);
  const mags = 'npc:mags-holloway';
  try {
    await timed('adaptation', () => game.start());
    const opening = (await timed('choice + opening scene', () => game.choose(0))) as TurnResult;
    if (!opening.reply) console.log('Empty opening:', opening.messages.map(describeMessage));
    for (const move of REFUSAL) await timed(move, () => game.send(move));
    console.log(table('Refusal', steps));

    // The refusal is in the world: Mags's ties or description changed (by the master or the generator).
    const refusalTurn = (await game.world.log.recent(200)).filter(
      (e) => e.turn >= 3 && ['relation.set', 'relation.cancelled', 'entity.edited'].includes(e.kind),
    );
    const touchesMags = refusalTurn.filter((e) => JSON.stringify(e.payload).includes(mags));
    console.log('\nWorld changes about Mags:', touchesMags.map((e) => `${e.actor}:${e.kind} ${JSON.stringify(e.payload).slice(0, 300)}`));
    const magsNow = await game.world.state.entity(mags);
    console.log('\nMags now:', magsNow?.description.slice(0, 600));
    console.log('Mags ties:', (await game.world.state.ties(mags)).map((t) => `${t.other.name}: ${t.description}`));
    console.log('Generator:', JSON.stringify(game.lastGeneration));
    const lines = await game.lines();
    console.log('\nTranscript after the refusal:');
    for (const line of lines.slice(-10)) console.log(`[${line.speaker}] ${line.text.slice(0, 400)}`);
    for (const who of [mags, 'npc:pip-quillon']) {
      console.log(`Actor notes of ${who}:`, (await game.world.log.actorNotes(who, 20)).map((n) => n.text));
    }
    // ...or in Mags's own memory: an actor note written on the refusal turn or later (6.2h)
    const magsNotes = (await game.world.log.recent(200)).filter(
      (e) => e.turn >= 3 && e.kind === 'note.added' && e.payload.entity === mags,
    );
    expect(touchesMags.length + magsNotes.length).toBeGreaterThan(0);
    // stage 6.2h: people speak in their own bubbles (the actor pack)
    expect(lines.some((l) => l.speaker.startsWith('npc:'))).toBe(true);
    const turns = steps.filter((s) => !s.step.startsWith('adaptation') && !s.step.startsWith('choice'));
    const average = turns.reduce((sum, s) => sum + s.cost, 0) / turns.length;
    console.log(`\nAverage cost of a player turn with the generator: $${average.toFixed(4)}; total ${formatUsage(game.sessionUsage)}`);
    console.log(`Refusal by channel:\n  ${formatChannels(game.sessionUsage).join('\n  ')}`);
  } finally {
    await game.close();
  }
});

it('stages 6.2j, 6.2k: luck, the Guard\'s recommendations and the generator (ADR 47, 54)', async () => {
  const game = await openNewGame('Luck');
  const steps: Step[] = [];
  const timed = recorder(game, steps);
  try {
    await timed('adaptation', () => game.start());
    await timed('choice + opening scene', () => game.choose(0));
    const before = new Set((await game.world.scene()).here.map((e) => e.id));
    for (const move of LUCK) await timed(move, () => game.send(move));
    console.log(table('Luck', steps));
    const events = await game.world.log.recent(100);
    const rolls = events.filter((e) => e.kind === 'luck.rolled');
    console.log('\nRolls:', rolls.map((e) => JSON.stringify(e.payload)));
    console.log('Master added:', events.filter((e) => e.actor === 'gm' && e.kind === 'entity.added').map((e) => JSON.stringify(e.payload)));
    console.log('Guard wrote (none since ADR 54):', events.filter((e) => e.actor === 'guard').map((e) => `${e.kind} ${JSON.stringify(e.payload).slice(0, 120)}`));
    console.log('Generator added:', events.filter((e) => e.actor === 'generator').map((e) => `${e.kind} ${JSON.stringify(e.payload).slice(0, 200)}`));
    const scene = await game.world.scene();
    console.log('New here:', scene.here.filter((e) => !before.has(e.id)).map((e) => e.name));
    console.log('Inventory:', scene.inventory.map((e) => e.name));
    const lines = await game.lines();
    for (const line of lines.slice(-6)) console.log(`[${line.speaker}] ${line.text.slice(0, 400)}`);
    expect(rolls.length).toBeGreaterThan(0);
  } finally {
    await game.close();
  }
});
