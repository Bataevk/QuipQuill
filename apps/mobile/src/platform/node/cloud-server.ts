/// <reference types="node" />
/**
 * A real cloud (FastAPI + RabbitMQ worker) for core integration tests: the echo handler instead
 * of the LLM, or (`live`) the real OpenRouter with the key from `cloud/.env` and the daily cap
 * raised (DAILY_COST_LIMIT_USD). Own queue prefix, so it does not disturb a running
 * `make cloud-dev`. Needs `make infra-up` and the cloud's uv environment.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { createServer } from 'node:net';
import { fileURLToPath } from 'node:url';

const CLOUD_DIR = fileURLToPath(new URL('../../../../../cloud', import.meta.url));
export const TEST_INVITE = 'TS-CORE-INVITE';

export interface TestCloud {
  httpUrl: string;
  wsUrl: string;
  stop(): Promise<void>;
}

export async function startTestCloud(
  options: { echoDelaySeconds?: number; live?: boolean; dailyLimitUsd?: number } = {},
): Promise<TestCloud> {
  const port = await freePort();
  const child = spawn(
    'uv',
    ['run', 'uvicorn', 'quipquill_cloud.main:app', '--host', '127.0.0.1', '--port', String(port)],
    {
      cwd: CLOUD_DIR,
      env: {
        ...process.env,
        ENV: 'test',
        DATABASE_URL:
          process.env.TEST_DATABASE_URL ??
          'postgresql+asyncpg://quipquill:quipquill@localhost:5432/quipquill_test',
        RABBITMQ_URL: process.env.TEST_RABBITMQ_URL ?? 'amqp://quipquill:quipquill@localhost:5672/',
        QUEUE_PREFIX: options.live ? 'qq.test.ts-live' : 'qq.test.ts-core',
        LLM_HANDLER: options.live ? 'openrouter' : 'echo',
        DAILY_COST_LIMIT_USD: String(options.dailyLimitUsd ?? 1),
        ECHO_DELAY_SECONDS: String(options.echoDelaySeconds ?? 0.01),
        INVITE_CODES: JSON.stringify([TEST_INVITE]),
        JWT_SECRET: 'ts-core-test-secret-that-is-long-enough-32b',
        WS_HEARTBEAT_INTERVAL: '0.5',
        LOG_LEVEL: 'warning',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  let output = '';
  child.stdout?.on('data', (chunk) => (output += String(chunk)));
  child.stderr?.on('data', (chunk) => (output += String(chunk)));
  const httpUrl = `http://127.0.0.1:${port}`;
  try {
    await waitForHealth(`${httpUrl}/health/mq`, child, 30_000);
  } catch (error) {
    child.kill('SIGKILL');
    throw new Error(`Cloud did not start: ${(error as Error).message}\n${output.slice(-2000)}`);
  }
  return {
    httpUrl,
    wsUrl: `ws://127.0.0.1:${port}/ws`,
    stop: () => stop(child),
  };
}

async function waitForHealth(url: string, child: ChildProcess, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`exited with ${child.exitCode}`);
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(1000) });
      if (response.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`no healthy answer from ${url} in ${timeoutMs} ms`);
}

function stop(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null) return Promise.resolve();
  return new Promise((resolve) => {
    const kill = setTimeout(() => child.kill('SIGKILL'), 5000);
    child.once('exit', () => {
      clearTimeout(kill);
      resolve();
    });
    child.kill('SIGTERM');
  });
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      server.close(() => (typeof address === 'object' && address ? resolve(address.port) : reject()));
    });
  });
}
