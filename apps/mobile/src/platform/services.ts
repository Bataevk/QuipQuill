/** Composition root: the core's ports wired to this platform's adapters. */
import { Platform } from 'react-native';

import { StoryCatalog } from '@/core/library/catalog';
import { Library } from '@/core/library/library';
import { AuthSession } from '@/core/net/auth';
import { CloudSocket } from '@/core/net/cloud-socket';
import { CloudEmbeddings } from '@/core/net/embeddings';
import { Game } from '@/core/game/game';
import { getAdapters } from './adapters';

let library: Promise<Library> | undefined;

/** The app-wide library (app.db stays open for the app's lifetime). */
export function getLibrary(): Promise<Library> {
  const { driver, files } = getAdapters();
  library ??= Library.open(driver, files).catch((error) => {
    library = undefined; // let the next call retry
    throw error;
  });
  return library;
}

/**
 * Cloud base URL: EXPO_PUBLIC_CLOUD_URL (e.g. http://192.168.1.10:8000 for a phone on the LAN).
 * Without it: the iOS simulator reaches the Mac's localhost directly, the Android emulator
 * reaches it as 10.0.2.2. A web build is served by the cloud's own host (ADR 69), so it talks to
 * the page's origin; the web dev server (Metro) is not the cloud, so dev uses localhost:8000.
 */
const LOCAL_CLOUD =
  Platform.OS === 'android'
    ? 'http://10.0.2.2:8000'
    : Platform.OS === 'web' && !__DEV__
      ? globalThis.location.origin
      : 'http://localhost:8000';
export const CLOUD_URL = (process.env.EXPO_PUBLIC_CLOUD_URL ?? LOCAL_CLOUD).replace(/\/+$/, '');

export interface Cloud {
  auth: AuthSession;
  socket: CloudSocket;
}

let cloud: Cloud | undefined;

let signingIn: Promise<void> | undefined;

/**
 * The prototype needs no sign-in (ADR 66): without tokens the device becomes a guest of the cloud
 * on its own, then the socket starts. Signing in with an invite (Profile) replaces the guest.
 * Fails quietly offline: the next call tries again.
 */
export function ensureSignedIn(): Promise<void> {
  signingIn ??= (async () => {
    const { auth, socket } = getCloud();
    if (!(await auth.isSignedIn())) await auth.guest();
    socket.start();
  })().finally(() => {
    signingIn = undefined;
  });
  return signingIn;
}

/** The signed-in session and the one WebSocket (call `socket.start()` once signed in). */
export function getCloud(): Cloud {
  if (!cloud) {
    const auth = new AuthSession(CLOUD_URL, getAdapters().tokens);
    const socket = new CloudSocket({ url: `${CLOUD_URL.replace(/^http/, 'ws')}/ws`, auth });
    cloud = { auth, socket };
  }
  return cloud;
}

let catalog: Promise<StoryCatalog> | undefined;

/** The cloud's story catalogue, downloading into this device's library (ADR 61). */
export function getCatalog(): Promise<StoryCatalog> {
  catalog ??= getLibrary().then(
    (library) =>
      new StoryCatalog({
        baseUrl: CLOUD_URL,
        auth: getCloud().auth,
        downloader: getAdapters().downloader,
        files: getAdapters().files,
        importer: library.importer,
        repository: library.repository,
      }),
  );
  return catalog;
}

/** Opens a playthrough wired to this platform's library and cloud (the game screen). */
export async function openGame(saveId: string): Promise<Game> {
  const { auth, socket } = getCloud();
  return Game.open(
    {
      library: await getLibrary(),
      socket,
      embeddings: (model) => new CloudEmbeddings(CLOUD_URL, auth, model),
    },
    saveId,
  );
}
