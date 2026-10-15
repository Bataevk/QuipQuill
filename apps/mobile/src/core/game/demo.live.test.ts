/**
 * Stage 6.1: the demo scenario end to end — this core, a local cloud, the real Haiku through
 * OpenRouter — three times in a row, with the cost of every move. Costs real money (about
 * $0.03 per run): `npm run test:live` after `make infra-up`.
 */
import { afterAll, beforeAll, expect, it } from 'vitest';

import { AuthSession, MemoryTokenStore } from '@/core/net/auth';
import { CloudSocket } from '@/core/net/cloud-socket';
import { CloudEmbeddings } from '@/core/net/embeddings';
import { formatUsage } from '@/core/protocol/usage-meter';
import { newId, nowIso } from '@/core/util/ids';
import { startTestCloud, TEST_INVITE, type TestCloud } from '@/platform/node/cloud-server';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import { DEMO_CHARACTER, DEMO_MOVES } from './demo-script';
import { Game } from './game';

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

async function playOnce(run: number): Promise<Step[]> {
  const characterId = newId();
  await env.library.repository.addCharacter({ id: characterId, ...DEMO_CHARACTER, createdAt: nowIso() });
  const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId, title: `Demo ${run}` });
  const game = await Game.open(
    { library: env.library, socket, embeddings: (model) => new CloudEmbeddings(cloud.httpUrl, auth, model) },
    save.id,
  );
  const steps: Step[] = [];
  const timed = async (step: string, action: () => Promise<{ reply?: string; events?: { kind: string }[] } | object>) => {
    const t0 = Date.now();
    const result = (await action()) as { reply?: string; events?: { kind: string }[]; rationale?: string };
    steps.push({
      step,
      ms: Date.now() - t0,
      cost: game.lastUsage?.costUsd ?? 0,
      usage: game.lastUsage ? formatUsage(game.lastUsage) : '',
      events: (result.events ?? []).map((e) => e.kind),
      reply: (result.reply ?? result.rationale ?? '').slice(0, 160).replace(/\s+/g, ' '),
    });
    return result;
  };
  try {
    const proposal = await timed('adaptation', () => game.start());
    expect((proposal as { options: unknown[] }).options.length).toBeGreaterThan(0);
    const opening = await timed('choice + opening scene', () => game.choose(0));
    expect(opening.reply).toBeTruthy();
    for (const move of DEMO_MOVES) {
      const turn = (await timed(move, () => game.send(move))) as { reply: string; cutShort: boolean };
      expect(turn.reply, move).toBeTruthy();
      expect(turn.cutShort, move).toBe(false);
    }
    expect(await game.nextStep()).toBeUndefined();
    const kinds = new Set(steps.flatMap((s) => s.events));
    console.log(`run ${run}: events ${[...kinds].join(', ')}; total ${formatUsage(game.sessionUsage)}`);
    return steps;
  } finally {
    await game.close();
  }
}

it(`plays the demo scenario ${RUNS} times in a row without errors`, async () => {
  for (let run = 1; run <= RUNS; run++) {
    const steps = await playOnce(run);
    console.log(
      [`\n### Run ${run}`, '| Step | s | $ | Usage | Events | Reply |', '|---|---|---|---|---|---|']
        .concat(
          steps.map(
            (s) =>
              `| ${s.step} | ${(s.ms / 1000).toFixed(1)} | ${s.cost.toFixed(4)} | ${s.usage} | ${s.events.join(' ') || '—'} | ${s.reply.replace(/\|/g, '/')} |`,
          ),
        )
        .join('\n'),
    );
  }
});
