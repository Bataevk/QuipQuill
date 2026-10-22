/**
 * The browser has every standard Web API the core relies on (ADR 27, ADR 69): `crypto` with
 * `randomUUID` (secure contexts: https and localhost), `navigator.userAgent`,
 * `AbortSignal.prototype.throwIfAborted`. `window.crypto` is read-only there, so the native
 * polyfills (polyfills.ts) must not run.
 *
 * langsmith has an import cycle (run_trees → address → utils/env → index → run_trees). When
 * LangChain reaches `langsmith/run_trees` first, the dev bundle's React Refresh reads
 * `index.RunTree` before run_trees has finished: "Cannot access 'RunTree' before
 * initialization" (web dev only, stage W.1). Loading the package root first closes the cycle in
 * a safe order.
 */
import 'langsmith';
