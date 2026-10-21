/** Downloader port on expo-file-system (SDK 57 `File.downloadFileAsync`). */
import { File } from 'expo-file-system';

import type { Downloader } from '@/core/net/download';
import { toUri } from './expo-files';

export class ExpoDownloader implements Downloader {
  async download(...[url, destination, options]: Parameters<Downloader['download']>): Promise<void> {
    await File.downloadFileAsync(url, new File(toUri(destination)), {
      headers: options.headers,
      idempotent: true, // replace a leftover from an interrupted download
      signal: options.signal,
      onProgress: options.onProgress
        ? ({ bytesWritten, totalBytes }) =>
            options.onProgress?.({ received: bytesWritten, total: totalBytes > 0 ? totalBytes : null })
        : undefined,
    });
  }
}
