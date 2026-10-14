// Hermes has no node:async_hooks (spike 3.1b, ADR 25); metro.config.js resolves the module here.
// LangChain/LangGraph only use AsyncLocalStorage to pass config implicitly; like their browser
// builds, we run without it, so config must be passed explicitly (getConfig() and interrupt()
// without arguments do not work). Same contract as MockAsyncLocalStorage in @langchain/core.
class AsyncLocalStorage {
  getStore() {
    return undefined;
  }

  run(_store, callback, ...args) {
    return callback(...args);
  }

  enterWith() {}

  disable() {}
}

module.exports = { AsyncLocalStorage };
