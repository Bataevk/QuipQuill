/**
 * Random ids. Uses the standard Web Crypto `crypto.randomUUID` (ADR 27): built into Node,
 * Electron and browsers; React Native gets it from src/platform/polyfills.ts (expo-crypto).
 */
export function newId(): string {
  const crypto = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (typeof crypto?.randomUUID !== 'function') {
    throw new Error('crypto.randomUUID is unavailable: load the platform polyfills first');
  }
  return crypto.randomUUID();
}

export function nowIso(): string {
  return new Date().toISOString();
}
