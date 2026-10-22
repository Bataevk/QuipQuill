import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Linking, StyleSheet, View } from 'react-native';

import { SavesNote } from '@/components/saves-note';
import { StoryImport } from '@/components/story-import';
import { TermsSheet } from '@/components/terms';
import { ThemedText } from '@/components/themed-text';
import { Button, Card, Chip, Screen } from '@/components/ui';
import { WaitlistForm } from '@/components/waitlist-form';
import { Spacing } from '@/constants/theme';
import type { Account } from '@/core/net/auth';
import { ensureSignedIn, getCloud } from '@/platform/services';
import { loadThemePreference, setThemePreference, type ThemePreference } from '@/platform/theme-preference';

const THEMES: { value: ThemePreference; label: string }[] = [
  { value: 'system', label: 'System' },
  { value: 'light', label: 'Parchment' },
  { value: 'dark', label: 'Night' },
];

const OFL_URL = 'https://openfontlicense.org';

/** Story import from a file: dev builds only, the demo story ships inside the app. */
const SHOW_IMPORT = __DEV__;

/** The profile (stage D.7): the account, the waitlist, the theme, the terms, the fonts' credits; story import in dev builds. */
export default function ProfileScreen() {
  const [account, setAccount] = useState<Account | null>();
  const [theme, setTheme] = useState<ThemePreference>('system');
  const [readingTerms, setReadingTerms] = useState(false);

  useFocusEffect(
    useCallback(() => {
      let active = true;
      void loadThemePreference().then((t) => active && setTheme(t));
      ensureSignedIn()
        .then(() => getCloud().auth.me())
        .then((a) => active && setAccount(a))
        .catch(() => active && setAccount(null));
      return () => {
        active = false;
      };
    }, []),
  );

  const chooseTheme = (value: ThemePreference) => {
    setTheme(value);
    void setThemePreference(value);
  };

  /** Back to being a guest of this device (ADR 66): a new guest, a fresh allowance. */
  const signOut = async () => {
    const { auth, socket } = getCloud();
    socket.stop();
    await auth.logout();
    setAccount(undefined);
    await ensureSignedIn().catch(() => {});
    setAccount(await auth.me().catch(() => null));
  };

  const used =
    account && account.totalLimitUsd ? Math.min(100, Math.round((account.spentUsd / account.totalLimitUsd) * 100)) : null;

  return (
    <Screen>
      <Card>
        <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
          ACCOUNT
        </ThemedText>
        <ThemedText type="label">
          {account === undefined ? '…' : account === null ? 'Offline' : (account.email ?? 'Guest of this device')}
        </ThemedText>
        {account === null ? (
          <ThemedText type="caption" themeColor="inkMuted">
            The server cannot be reached right now.
          </ThemedText>
        ) : null}
        {used !== null ? (
          <ThemedText type="caption" themeColor="inkMuted">
            Prototype allowance: {used}% used.
          </ThemedText>
        ) : null}
        {account?.kind === 'email' ? (
          <Button title="Sign out" variant="ghost" onPress={() => void signOut()} />
        ) : (
          <Button title="Sign in with an invite" variant="ghost" onPress={() => router.push('/login')} />
        )}
      </Card>

      <SavesNote />

      <Card>
        <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
          THE FULL VERSION
        </ThemedText>
        <WaitlistForm source="profile" intro="This is a prototype. Leave an email to hear when foolore is released." />
      </Card>

      <Card>
        <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
          THEME
        </ThemedText>
        <View style={styles.chips}>
          {THEMES.map((t) => (
            <Chip key={t.value} label={t.label} selected={theme === t.value} onPress={() => chooseTheme(t.value)} />
          ))}
        </View>
      </Card>

      <Card>
        <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
          TERMS
        </ThemedText>
        <ThemedText type="caption" themeColor="inkMuted">
          You accepted the draft terms when you started.
        </ThemedText>
        <Button title="Read the terms and privacy" variant="ghost" onPress={() => setReadingTerms(true)} />
      </Card>
      <TermsSheet visible={readingTerms} onClose={() => setReadingTerms(false)} />

      <Card>
        <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
          ACKNOWLEDGEMENTS
        </ThemedText>
        <ThemedText type="caption">
          IM Fell English © 2010 Igino Marini. Literata © 2017 The Literata Project Authors. Both fonts are used
          under the SIL Open Font License 1.1.
        </ThemedText>
        <Button title="Read the license" variant="ghost" onPress={() => void Linking.openURL(OFL_URL)} />
      </Card>

      {SHOW_IMPORT ? <StoryImport /> : null}
    </Screen>
  );
}

const styles = StyleSheet.create({
  eyebrow: { letterSpacing: 1.2 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.two },
});
