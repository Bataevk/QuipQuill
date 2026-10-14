import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

// Core tests run in plain Node (ADR: vitest + node:sqlite, stage 3.2). No React Native here.
export default defineConfig({
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
