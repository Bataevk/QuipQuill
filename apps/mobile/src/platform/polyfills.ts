/**
 * Standard Web APIs the core relies on and React Native lacks. Import first, before anything
 * that loads the core (src/app/_layout.tsx). ADR 27: the core only uses `globalThis.crypto`;
 * here it is backed by expo-crypto (system CSPRNG) on iOS and Android.
 */
import { getRandomValues, randomUUID } from 'expo-crypto';

type WebCrypto = {
  getRandomValues?: <T extends ArrayBufferView>(array: T) => T;
  randomUUID?: () => string;
};

const g = globalThis as { crypto?: WebCrypto };
const crypto: WebCrypto = g.crypto ?? {};
if (typeof crypto.randomUUID !== 'function') crypto.randomUUID = randomUUID;
if (typeof crypto.getRandomValues !== 'function') {
  crypto.getRandomValues = <T extends ArrayBufferView>(array: T): T => {
    getRandomValues(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
    return array;
  };
}
g.crypto = crypto;
