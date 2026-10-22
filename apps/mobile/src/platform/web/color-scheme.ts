/**
 * The player's theme choice in the browser (stage W.3): react-native-web's Appearance only reads
 * the system's `prefers-color-scheme` and has no setColorScheme, so the choice lives here and
 * use-color-scheme.web.ts lays it over the system's.
 */
export type SchemeOverride = 'light' | 'dark' | null;

let override: SchemeOverride = null;
const listeners = new Set<() => void>();

export const getSchemeOverride = (): SchemeOverride => override;

export function setSchemeOverride(value: SchemeOverride): void {
  override = value;
  for (const listener of listeners) listener();
}

export function subscribeSchemeOverride(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
