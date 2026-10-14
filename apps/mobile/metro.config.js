// Learn more https://docs.expo.dev/guides/customizing-metro
const path = require('node:path');
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Spike 3.1b: LangChain imports node:async_hooks, which Hermes lacks (see the shim).
const nodeShims = {
  'node:async_hooks': path.resolve(__dirname, 'spikes/shims/async-hooks.js'),
  async_hooks: path.resolve(__dirname, 'spikes/shims/async-hooks.js'),
};
const resolveDefault = config.resolver.resolveRequest;
config.resolver.resolveRequest = (context, moduleName, platform) => {
  if (nodeShims[moduleName]) {
    return { type: 'sourceFile', filePath: nodeShims[moduleName] };
  }
  return (resolveDefault ?? context.resolveRequest)(context, moduleName, platform);
};

module.exports = config;
