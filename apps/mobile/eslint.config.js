// https://docs.expo.dev/guides/using-eslint/
const { defineConfig, globalIgnores } = require('eslint/config');
const expoConfig = require('eslint-config-expo/flat');

// src/core is plain TypeScript: it runs in Node tests and will move to desktop.
// Platform code reaches it only through ports (see apps/mobile/CLAUDE.md).
const coreBoundary = {
  files: ['src/core/**/*.{ts,tsx}'],
  rules: {
    'no-restricted-imports': [
      'error',
      {
        patterns: [
          {
            group: [
              'react',
              'react-native',
              'react-native-*',
              'expo',
              'expo-*',
              '@expo/*',
              '@op-engineering/*',
            ],
            message: 'src/core must not depend on React Native or Expo; use a port instead.',
          },
          {
            group: ['@/*', '!@/core', '!@/core/*'],
            message: 'src/core must not import app code (screens, components, platform adapters).',
          },
        ],
      },
    ],
  },
};

module.exports = defineConfig([globalIgnores(['dist/*']), expoConfig, coreBoundary]);
