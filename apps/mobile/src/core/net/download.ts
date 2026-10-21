/** File download port (stage D.5): the platform saves a URL to a local file, with progress. */

export interface DownloadProgress {
  received: number;
  /** Null when the server did not say how big the file is. */
  total: number | null;
}

export interface Downloader {
  /**
   * Saves `url` to the absolute path `destination`, replacing it. Rejects on a network error or
   * when `signal` aborts; the caller checks what arrived (the size it expects).
   */
  download(
    url: string,
    destination: string,
    options: { headers?: Record<string, string>; onProgress?: (progress: DownloadProgress) => void; signal?: AbortSignal },
  ): Promise<void>;
}
