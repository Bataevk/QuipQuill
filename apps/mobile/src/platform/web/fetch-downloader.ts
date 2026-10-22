/** Downloader port on fetch, saving into the browser's storage (stage W.2, ADR 69). */
import type { Downloader } from '@/core/net/download';
import type { OpfsFileStore } from './opfs-files';

export class FetchDownloader implements Downloader {
  private readonly files: OpfsFileStore;

  constructor(files: OpfsFileStore) {
    this.files = files;
  }

  async download(...[url, destination, options]: Parameters<Downloader['download']>): Promise<void> {
    const response = await fetch(url, { headers: options.headers, signal: options.signal });
    if (!response.ok) {
      // Nothing saved, as when the phone's download is refused: the catalogue reads the missing
      // file as a refused token and refreshes it once.
      await this.files.remove(destination);
      return;
    }
    const header = Number(response.headers.get('content-length'));
    const total = Number.isFinite(header) && header > 0 ? header : null;
    const chunks: Uint8Array[] = [];
    let received = 0;
    const reader = response.body?.getReader();
    if (reader) {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.byteLength;
        options.onProgress?.({ received, total });
      }
    } else {
      chunks.push(new Uint8Array(await response.arrayBuffer()));
      received = chunks[0].byteLength;
      options.onProgress?.({ received, total });
    }
    const bytes = new Uint8Array(received);
    let at = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, at);
      at += chunk.byteLength;
    }
    // Whatever came is saved; the catalogue checks the size it expects.
    await this.files.write(destination, bytes);
  }
}
