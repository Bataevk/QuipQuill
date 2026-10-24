// Learn more https://docs.expo.dev/guides/customizing-metro
const path = require('node:path');
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// LangChain imports node:async_hooks, which Hermes lacks (ADR 25; see the shim).
const asyncHooksShim = path.resolve(__dirname, 'src/platform/shims/async-hooks.js');
const nodeShims = {
  'node:async_hooks': asyncHooksShim,
  async_hooks: asyncHooksShim,
};
const resolveDefault = config.resolver.resolveRequest;
config.resolver.resolveRequest = (context, moduleName, platform) => {
  if (nodeShims[moduleName]) {
    return { type: 'sourceFile', filePath: nodeShims[moduleName] };
  }
  return (resolveDefault ?? context.resolveRequest)(context, moduleName, platform);
};

// The demo story ships inside the app for now (stories/alice since ADR 80, built by builder): Metro
// bundles it as an asset from the repo's stories/ folder, the single copy of the file.
config.watchFolders = [...(config.watchFolders ?? []), path.resolve(__dirname, '../../stories')];
config.resolver.assetExts.push('qqstory');

module.exports = config;
