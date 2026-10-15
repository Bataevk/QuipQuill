import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

// Core tests run in plain Node (ADR 28: vitest + node:sqlite). No React Native here.
// *.int.test.ts talk to a local cloud (make infra-up): only with QQ_CLOUD=1 (npm run test:cloud).
// *.live.test.ts call the real model through it and cost money: only with QQ_LIVE=1
// (npm run test:live).
const withCloud = process.env.QQ_CLOUD === '1';
const live = process.env.QQ_LIVE === '1';

export default defineConfig({
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  test: {
    environment: 'node',
    include: live ? ['src/**/*.live.test.ts'] : withCloud ? ['src/**/*.int.test.ts'] : ['src/**/*.test.ts'],
    exclude: live || withCloud ? [] : ['src/**/*.int.test.ts', 'src/**/*.live.test.ts'],
    testTimeout: live ? 20 * 60_000 : withCloud ? 30_000 : 5_000,
  },
});
