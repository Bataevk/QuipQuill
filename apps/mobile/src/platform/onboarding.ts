/**
 * The first launch (stage D.6): whether it is done, and the terms the player accepted (ADR 60).
 * Kept in app.db settings; the consent also goes to the cloud when it can be reached.
 */
import { ensureSignedIn, getCloud, getLibrary } from './services';

/** The stub terms (ADR 60): a draft until a lawyer writes the real ones (backlog). */
export const TERMS_VERSION = '0-draft';

/** Done, or the app was in use before onboarding existed (it already has games). */
export async function isOnboarded(): Promise<boolean> {
  const { repository } = await getLibrary();
  if ((await repository.getSetting('onboarded')) === '1') return true;
  return (await repository.listSaves()).length > 0;
}

export async function finishOnboarding(): Promise<void> {
  await (await getLibrary()).repository.setSetting('onboarded', '1');
}

/** Kept on the device first; the cloud's copy follows, or waits for the next acceptance offline. */
export async function acceptTerms(): Promise<void> {
  const { repository } = await getLibrary();
  await repository.setSetting('terms_version', TERMS_VERSION);
  await repository.setSetting('terms_accepted_at', new Date().toISOString());
  void ensureSignedIn()
    .then(() => getCloud().auth.consent(TERMS_VERSION))
    .catch(() => {});
}
