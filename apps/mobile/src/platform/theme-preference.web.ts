/**
 * The player's theme choice (Profile, stage D.7) in the browser (stage W.3): kept in app.db
 * settings as on the phone, applied through the override that use-color-scheme.web.ts reads, and
 * to the page's `color-scheme`, so scrollbars and form controls follow it too.
 */
import { getLibrary } from './services';
import { setSchemeOverride } from './web/color-scheme';

export type ThemePreference = 'system' | 'light' | 'dark';

const KEY = 'theme';

const parse = (value: string | undefined): ThemePreference =>
  value === 'light' || value === 'dark' ? value : 'system';

function apply(preference: ThemePreference) {
  setSchemeOverride(preference === 'system' ? null : preference);
  if (typeof document !== 'undefined') {
    document.documentElement.style.colorScheme = preference === 'system' ? 'light dark' : preference;
  }
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
