/** FileStore port on expo-file-system (SDK 57 object API). Plain paths in, file:// URIs inside. */
import { Directory, File, Paths } from 'expo-file-system';

import type { FileStore } from '@/core/storage/files';

export class ExpoFileStore implements FileStore {
  readonly dataDir: string;

  constructor(dataDir: string = toPath(Paths.document.uri)) {
    this.dataDir = dataDir.replace(/\/+$/, '');
  }

  async exists(path: string): Promise<boolean> {
    return new File(toUri(path)).exists || new Directory(toUri(path)).exists;
  }

  async copy(from: string, to: string): Promise<void> {
    const target = new File(toUri(to));
    if (target.exists) target.delete();
    await new File(toUri(from)).copy(target);
  }

  async move(from: string, to: string): Promise<void> {
    const target = new File(toUri(to));
    if (target.exists) target.delete();
    await new File(toUri(from)).move(target);
  }

  async remove(path: string): Promise<void> {
    const file = new File(toUri(path));
    if (file.exists) file.delete();
  }

  async ensureDir(path: string): Promise<void> {
    const dir = new Directory(toUri(path));
    if (!dir.exists) dir.create({ intermediates: true });
  }
}

/** file:///a%20b/c → /a b/c */
export function toPath(uri: string): string {
  return uri.startsWith('file://') ? decodeURIComponent(uri.slice('file://'.length)) : uri;
}

/** /a b/c → file:///a%20b/c */
export function toUri(path: string): string {
  return path.startsWith('file://') ? path : `file://${encodeURI(path)}`;
}

/**
 * A picked file as a plain path the core can open. iOS hands out a file:// copy already;
 * Android's picker gives a content:// URI, so that one is copied into the cache first
 * (`temporary`: delete it after use).
 */
export async function pickedFilePath(file: File): Promise<{ path: string; temporary: boolean }> {
  if (file.uri.startsWith('file://')) return { path: toPath(file.uri), temporary: false };
  const copy = new File(Paths.cache, `import-${Date.now()}.qqstory`);
  await file.copy(copy);
  return { path: toPath(copy.uri), temporary: true };
}

export function removeFile(path: string): void {
  const file = new File(toUri(path));
  if (file.exists) file.delete();
}
