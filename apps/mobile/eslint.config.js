// https://docs.expo.dev/guides/using-eslint/
const { defineConfig, globalIgnores } = require('eslint/config');
const expoConfig = require('eslint-config-expo/flat');

// src/core is plain TypeScript: it runs in Node tests and will move to desktop.
// Platform code reaches it only through ports (see apps/mobile/CLAUDE.md).
const noReactNative = {
  group: ['react', 'react-native', 'react-native-*', 'expo', 'expo-*', '@expo/*', '@op-engineering/*'],
  message: 'src/core must not depend on React Native or Expo; use a port instead.',
};
const noAppCode = (allowed) => ({
  group: ['@/*', '!@/core', '!@/core/*', ...allowed],
  message: 'src/core must not import app code (screens, components, platform adapters).',
});
const coreBoundary = (files, ignores, allowed) => ({
  files,
  ignores,
  rules: {
    'no-restricted-imports': ['error', { patterns: [noReactNative, noAppCode(allowed)] }],
  },
});

module.exports = defineConfig([
  globalIgnores(['dist/*', 'public/sqlite/sqlite3.*']), // sqlite3.* are copies from node_modules
  expoConfig,
  coreBoundary(['src/core/**/*.{ts,tsx}'], ['src/core/**/*.test.ts'], []),
  // Core tests run in Node and may use the Node adapters (node:sqlite, node:fs).
  coreBoundary(['src/core/**/*.test.ts'], [], ['!@/platform', '@/platform/*', '!@/platform/node']),
]);
