/**
 * The demo story ships inside the app for now (owner, 2026-10-20), until stories come from the
 * cloud catalogue (D.4–D.5): the player gets it with no file to import. Metro bundles the
 * builder's file from the repo (metro.config.js); a new build of it replaces the imported copy,
 * saves keep their own copies (ADR 24).
 */
import { Asset } from 'expo-asset';

import { toPath } from './expo-files';
import { getLibrary } from './services';

// eslint-disable-next-line @typescript-eslint/no-require-imports
const DEMO_STORY = require('../../../../stories/brass-vault/brass-vault.qqstory');

let ready: Promise<void> | undefined;

/** Imports the shipped story if the library lacks it or holds another build; once per launch. */
export function ensureBundledStory(): Promise<void> {
  ready ??= (async () => {
    const asset = await Asset.fromModule(DEMO_STORY).downloadAsync();
    if (!asset.localUri) throw new Error('The built-in story could not be opened.');
    await (await getLibrary()).importer.importIfChanged(toPath(asset.localUri));
  })().catch((error: unknown) => {
    ready = undefined; // let the next visit try again
    throw error;
  });
  return ready;
}
