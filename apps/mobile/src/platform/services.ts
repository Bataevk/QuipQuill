/** Composition root: the core's ports wired to this platform's adapters. */
import { Library } from '@/core/library/library';
import { AuthSession } from '@/core/net/auth';
import { CloudSocket } from '@/core/net/cloud-socket';
import { CloudEmbeddings } from '@/core/net/embeddings';
import { Game } from '@/core/game/game';
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
 * Cloud base URL: EXPO_PUBLIC_CLOUD_URL (e.g. http://192.168.1.10:8000 for a phone on the LAN);
 * the iOS simulator reaches the Mac's localhost directly.
 */
export const CLOUD_URL = (process.env.EXPO_PUBLIC_CLOUD_URL ?? 'http://localhost:8000').replace(/\/+$/, '');

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
