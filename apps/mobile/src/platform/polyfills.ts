/**
 * Standard Web APIs the core relies on and React Native lacks. Import first, before anything
 * that loads the core (src/app/_layout.tsx). ADR 27: the core only uses `globalThis.crypto`;
 * here it is backed by expo-crypto (system CSPRNG) on iOS and Android. ADR 25: the rest is what
 * LangChain needs on Hermes (spike 3.1b) — recheck on every `langchain` / `@langchain/*` update.
 */
import { getRandomValues, randomUUID } from 'expo-crypto';

type WebCrypto = {
  getRandomValues?: <T extends ArrayBufferView>(array: T) => T;
  randomUUID?: () => string;
};

const g = globalThis as { crypto?: WebCrypto; navigator?: { userAgent?: string } };
const crypto: WebCrypto = g.crypto ?? {};
if (typeof crypto.randomUUID !== 'function') crypto.randomUUID = randomUUID;
if (typeof crypto.getRandomValues !== 'function') {
  crypto.getRandomValues = <T extends ArrayBufferView>(array: T): T => {
    getRandomValues(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
    return array;
  };
}
g.crypto = crypto;

// @langchain/core's isJsDom() calls navigator.userAgent.includes().
if (g.navigator && typeof g.navigator.userAgent !== 'string') g.navigator.userAgent = 'ReactNative';

// AbortSignal.prototype.throwIfAborted — LangChain calls it on every step of a run.
const abortProto = globalThis.AbortSignal?.prototype as
  | (AbortSignal & { throwIfAborted?: () => void })
  | undefined;
if (abortProto && typeof abortProto.throwIfAborted !== 'function') {
  abortProto.throwIfAborted = function throwIfAborted(this: AbortSignal) {
    if (this.aborted) throw this.reason ?? new Error('This operation was aborted');
  };
}
