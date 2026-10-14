// Spike 3.1b: Hermes has no node:async_hooks. LangChain/LangGraph only use AsyncLocalStorage
// to pass config implicitly; like their browser builds, we run without it (config is passed
// explicitly). Same contract as MockAsyncLocalStorage in @langchain/core.
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
