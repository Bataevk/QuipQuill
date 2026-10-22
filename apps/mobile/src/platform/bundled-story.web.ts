/**
 * The built-in story in the browser (stage W.2, ADR 69): as on the phone (bundled-story.ts), but
 * the asset is a URL, so its bytes are fetched into the storage pool first and imported from
 * there; the staging copy is removed afterwards.
 */
import { Asset } from 'expo-asset';

import { getAdapters } from './adapters.web';
import { getLibrary } from './services';

// eslint-disable-next-line @typescript-eslint/no-require-imports
const DEMO_STORY = require('../../../../stories/brass-vault/brass-vault.qqstory');

/** Per launch: the pool has no directories, only names. */
const STAGING = '/qq/bundled/brass-vault.qqstory';
const FETCH_TIMEOUT_MS = 30_000;

let ready: Promise<void> | undefined;

/** Imports the shipped story if the library lacks it or holds another build; once per launch. */
export function ensureBundledStory(): Promise<void> {
  ready ??= (async () => {
    const response = await fetch(Asset.fromModule(DEMO_STORY).uri, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
    if (!response.ok) throw new Error('The built-in story could not be opened.');
    const { files } = getAdapters();
    await files.write(STAGING, new Uint8Array(await response.arrayBuffer()));
    try {
      await (await getLibrary()).importer.importIfChanged(STAGING);
    } finally {
      await files.remove(STAGING);
    }
  })().catch((error: unknown) => {
    ready = undefined; // let the next visit try again
    throw error;
  });
  return ready;
}
