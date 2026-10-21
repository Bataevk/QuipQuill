/** Composition root: the core's ports wired to this platform's adapters. */
import { Platform } from 'react-native';

import { StoryCatalog } from '@/core/library/catalog';
import { Library } from '@/core/library/library';
import { AuthSession } from '@/core/net/auth';
import { CloudSocket } from '@/core/net/cloud-socket';
import { CloudEmbeddings } from '@/core/net/embeddings';
import { Game } from '@/core/game/game';
import { ExpoDownloader } from './expo-downloader';
import { ExpoFileStore } from './expo-files';
import { OpSqliteDriver } from './op-sqlite-driver';
import { SecureTokenStore } from './secure-token-store';

let library: Promise<Library> | undefined;

/** The app-wide library (app.db stays open for the app's lifetime). */
export function getLibrary(): Promise<Library> {
  library ??= Library.open(new OpSqliteDriver(), new ExpoFileStore()).catch((error) => {
    library = undefined; // let the next call retry
    throw error;
  });
  return library;
}

/**
 * Cloud base URL: EXPO_PUBLIC_CLOUD_URL (e.g. http://192.168.1.10:8000 for a phone on the LAN).
 * Without it: the iOS simulator reaches the Mac's localhost directly, the Android emulator
 * reaches it as 10.0.2.2.
 */
const LOCAL_CLOUD = Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000';
export const CLOUD_URL = (process.env.EXPO_PUBLIC_CLOUD_URL ?? LOCAL_CLOUD).replace(/\/+$/, '');

export interface Cloud {
  auth: AuthSession;
  socket: CloudSocket;
}

let cloud: Cloud | undefined;

/** The signed-in session and the one WebSocket (call `socket.start()` once signed in). */
export function getCloud(): Cloud {
  if (!cloud) {
    const auth = new AuthSession(CLOUD_URL, new SecureTokenStore());
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
        downloader: new ExpoDownloader(),
        files: new ExpoFileStore(),
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
