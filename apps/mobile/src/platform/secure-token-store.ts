/** TokenStore port on expo-secure-store (Keychain on iOS, Keystore-backed on Android). */
import { deleteItemAsync, getItemAsync, setItemAsync } from 'expo-secure-store';

import type { Tokens, TokenStore } from '@/core/net/auth';

const KEY = 'quipquill.tokens';

export class SecureTokenStore implements TokenStore {
  async load(): Promise<Tokens | null> {
    const raw = await getItemAsync(KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw) as Tokens;
    } catch {
      await deleteItemAsync(KEY); // corrupt entry: sign in again
      return null;
    }
  }

  async save(tokens: Tokens): Promise<void> {
    await setItemAsync(KEY, JSON.stringify(tokens));
  }

  async clear(): Promise<void> {
    await deleteItemAsync(KEY);
  }
}
