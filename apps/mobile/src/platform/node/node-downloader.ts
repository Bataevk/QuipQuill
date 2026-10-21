/// <reference types="node" />
/** Downloader port on fetch + node:fs (core tests against the local cloud). */
import { writeFile } from 'node:fs/promises';

import type { Downloader } from '@/core/net/download';

export class NodeDownloader implements Downloader {
  async download(...[url, destination, options]: Parameters<Downloader['download']>): Promise<void> {
    const response = await fetch(url, { headers: options.headers, signal: options.signal });
    const bytes = new Uint8Array(await response.arrayBuffer());
    // Like the phone's downloader: whatever came is saved; the catalogue checks the size.
    await writeFile(destination, response.ok ? bytes : new Uint8Array());
    options.onProgress?.({ received: bytes.length, total: bytes.length });
  }
}
