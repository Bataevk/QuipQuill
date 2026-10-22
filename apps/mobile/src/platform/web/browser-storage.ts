/**
 * The browser's storage (stage W.2, ADR 69): SQLite WASM on OPFS in a worker
 * (public/sqlite/db-worker.mjs), reached by messages. One tab at a time: opfs-sahpool holds one
 * connection, so a second tab gets StorageBusyError instead of a broken pool (spike W.1).
 */
import type { StorageRequest, StorageResults } from '../../../public/sqlite/db-host.mjs';

/** What the client needs from a Worker; tests connect straight to the host. */
export interface WorkerPort {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: { data: unknown }) => void) | null;
  onerror: ((event: { message?: string }) => void) | null;
}

/** Another tab of the app holds the storage. */
export class StorageBusyError extends Error {
  constructor() {
    super('foolore is already open in another tab. Close it there, then reload this page.');
    this.name = 'StorageBusyError';
  }
}

type Reply = { id: number; result?: unknown; error?: { name: string; message: string } };
type Pending = { resolve: (value: unknown) => void; reject: (error: Error) => void };

export class BrowserStorage {
  private readonly connect: () => Promise<WorkerPort>;
  private port: Promise<WorkerPort> | undefined;
  private readonly pending = new Map<number, Pending>();
  private nextId = 1;
  private broken: Error | undefined;

  constructor(connect: () => Promise<WorkerPort>) {
    this.connect = connect;
  }

  async call<K extends StorageRequest['op']>(
    request: Extract<StorageRequest, { op: K }>,
    transfer?: Transferable[],
  ): Promise<StorageResults[K]> {
    const port = await this.start();
    if (this.broken) throw this.broken;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
      port.postMessage({ id, request }, transfer);
    });
  }

  private start(): Promise<WorkerPort> {
    this.port ??= this.connect()
      .then((port) => {
        port.onmessage = ({ data }) => this.settle(data as Reply);
        port.onerror = (event) => this.fail(new Error(`The storage worker failed: ${event.message ?? 'unknown error'}`));
        return port;
      })
      .catch((error: unknown) => {
        this.port = undefined; // a busy tab may be closed meanwhile: let the next call try again
        throw error;
      });
    return this.port;
  }

  private settle({ id, result, error }: Reply) {
    const pending = this.pending.get(id);
    if (!pending) return;
    this.pending.delete(id);
    if (!error) {
      pending.resolve(result);
      return;
    }
    const failure = new Error(error.message);
    failure.name = error.name;
    pending.reject(failure);
  }

  private fail(error: Error) {
    this.broken = error;
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
}

const LOCK = 'foolore-storage';

/**
 * The real worker, once this tab holds the storage lock (Web Locks; kept until the tab closes).
 * Also asks the browser to keep the site's data: Safari otherwise clears it after 7 days
 * without a visit (ADR 39).
 */
export async function connectWorker(): Promise<WorkerPort> {
  if (!(await holdTabLock())) throw new StorageBusyError();
  void navigator.storage?.persist?.().catch(() => false);
  return new Worker('/sqlite/db-worker.mjs', { type: 'module' }) as unknown as WorkerPort;
}

function holdTabLock(): Promise<boolean> {
  const locks = navigator.locks;
  if (!locks) return Promise.resolve(true); // very old browser: no way to tell, try anyway
  return new Promise((resolve) => {
    void locks.request(LOCK, { ifAvailable: true }, (lock) => {
      resolve(lock !== null);
      return lock ? new Promise<never>(() => {}) : undefined;
    });
  });
}
