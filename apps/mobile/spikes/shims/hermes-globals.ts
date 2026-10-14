// Spike 3.1b: globals that LangChain expects and Hermes / React Native lack.
// Minimal versions, only to get the spike running; the findings go to docs/notes/spikes.md.

type AnyFn = (...args: never[]) => unknown;
const g = globalThis as Record<string, unknown> & {
  crypto?: { getRandomValues?: <T extends ArrayBufferView>(a: T) => T };
  navigator?: { userAgent?: string };
};
const abortProto = (globalThis.AbortSignal?.prototype ?? {}) as unknown as Record<string, unknown>;
const AbortSignalCtor = globalThis.AbortSignal as unknown as Record<string, unknown> | undefined;

/** What the runtime has before any patching (reported by the spike screen). */
export const nativeGlobals = {
  cryptoGetRandomValues: typeof g.crypto?.getRandomValues === 'function',
  navigatorUserAgent: typeof g.navigator?.userAgent === 'string',
  abortSignalThrowIfAborted: typeof abortProto.throwIfAborted === 'function',
  abortSignalAny: typeof AbortSignalCtor?.any === 'function',
  abortSignalTimeout: typeof AbortSignalCtor?.timeout === 'function',
  structuredClone: typeof g.structuredClone === 'function',
  readableStream: typeof g.ReadableStream === 'function',
  textEncoder: typeof g.TextEncoder === 'function',
  textDecoder: typeof g.TextDecoder === 'function',
  textDecoderStream: typeof g.TextDecoderStream === 'function',
};

// crypto.getRandomValues — LangGraph's uuid needs it. NOT cryptographically secure: the app
// needs a real polyfill (e.g. expo-crypto) before 3.4, where request ids are uuids too.
if (!nativeGlobals.cryptoGetRandomValues) {
  g.crypto = {
    ...g.crypto,
    getRandomValues<T extends ArrayBufferView>(array: T): T {
      const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
      for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
      return array;
    },
  };
}

// @langchain/core's isJsDom() calls navigator.userAgent.includes().
if (g.navigator && !nativeGlobals.navigatorUserAgent) {
  g.navigator.userAgent = 'ReactNative';
}

// AbortSignal.prototype.throwIfAborted — called on every step of a LangChain stream.
if (globalThis.AbortSignal && !nativeGlobals.abortSignalThrowIfAborted) {
  abortProto.throwIfAborted = function throwIfAborted(this: AbortSignal) {
    if (this.aborted) throw this.reason ?? new Error('This operation was aborted');
  };
}

// AbortSignal.any — combines the caller's signal with internal ones.
if (AbortSignalCtor && !nativeGlobals.abortSignalAny) {
  AbortSignalCtor.any = (signals: AbortSignal[]) => {
    const controller = new AbortController();
    for (const signal of signals) {
      if (signal.aborted) {
        controller.abort(signal.reason);
        break;
      }
      signal.addEventListener('abort', () => controller.abort(signal.reason));
    }
    return controller.signal;
  };
}

// AbortSignal.timeout.
if (AbortSignalCtor && !nativeGlobals.abortSignalTimeout) {
  AbortSignalCtor.timeout = (ms: number) => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(new Error('TimeoutError')), ms);
    return controller.signal;
  };
}

// structuredClone — LangGraph copies state; our state is plain JSON-like data plus messages.
if (!nativeGlobals.structuredClone) {
  (g as { structuredClone?: AnyFn }).structuredClone = ((value: unknown) =>
    value === undefined ? undefined : JSON.parse(JSON.stringify(value))) as AnyFn;
}
