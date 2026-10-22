/**
 * TokenStore port on localStorage (stage W.2, ADR 69). The browser has no keychain; the tokens
 * belong to a guest of this browser (ADR 66) and live as long as the site's data.
 */
import type { Tokens, TokenStore } from '@/core/net/auth';

const KEY = 'quipquill.tokens';

export class LocalTokenStore implements TokenStore {
  async load(): Promise<Tokens | null> {
    const raw = read();
    if (!raw) return null;
    try {
      return JSON.parse(raw) as Tokens;
    } catch {
      await this.clear(); // corrupt entry: a new guest takes over
      return null;
    }
  }

  async save(tokens: Tokens): Promise<void> {
    localStorage.setItem(KEY, JSON.stringify(tokens));
  }

  async clear(): Promise<void> {
    try {
      localStorage.removeItem(KEY);
    } catch {
      // storage blocked: nothing to clear
    }
  }
}

function read(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null; // storage blocked (private mode, site data off): sign in as a guest each time
  }
}
