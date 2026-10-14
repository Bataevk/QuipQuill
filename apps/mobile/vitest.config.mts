import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

// Core tests run in plain Node (ADR 28: vitest + node:sqlite). No React Native here.
// *.int.test.ts talk to a local cloud (make infra-up): only with QQ_CLOUD=1 (npm run test:cloud).
const withCloud = process.env.QQ_CLOUD === '1';

export default defineConfig({
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  test: {
    environment: 'node',
    include: withCloud ? ['src/**/*.int.test.ts'] : ['src/**/*.test.ts'],
    exclude: withCloud ? [] : ['src/**/*.int.test.ts'],
    testTimeout: withCloud ? 30_000 : 5_000,
  },
});
