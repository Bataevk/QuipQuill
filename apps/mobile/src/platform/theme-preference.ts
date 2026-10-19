/**
 * The player's theme choice (Profile, stage D.7): follow the system, or always Parchment / Night.
 * Kept in app.db settings and applied through Appearance, so `useColorScheme`, the keyboard and
 * system alerts all follow it.
 */
import { Appearance } from 'react-native';

import { getLibrary } from './services';

export type ThemePreference = 'system' | 'light' | 'dark';

const KEY = 'theme';

const parse = (value: string | undefined): ThemePreference =>
  value === 'light' || value === 'dark' ? value : 'system';

function apply(preference: ThemePreference) {
  Appearance.setColorScheme(preference === 'system' ? 'unspecified' : preference);
}

export async function loadThemePreference(): Promise<ThemePreference> {
  return parse(await (await getLibrary()).repository.getSetting(KEY));
}

/** At start-up: before the first screen settles, so it does not flash in the wrong theme. */
export async function restoreThemePreference(): Promise<void> {
  apply(await loadThemePreference());
}

export async function setThemePreference(preference: ThemePreference): Promise<void> {
  apply(preference);
  await (await getLibrary()).repository.setSetting(KEY, preference);
}
