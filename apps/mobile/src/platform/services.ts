/** Composition root: the core's ports wired to this platform's adapters. */
import { Library } from '@/core/library/library';
import { ExpoFileStore } from './expo-files';
import { OpSqliteDriver } from './op-sqlite-driver';

let library: Promise<Library> | undefined;

/** The app-wide library (app.db stays open for the app's lifetime). */
export function getLibrary(): Promise<Library> {
  library ??= Library.open(new OpSqliteDriver(), new ExpoFileStore()).catch((error) => {
    library = undefined; // let the next call retry
    throw error;
  });
  return library;
}
