import { useEffect, useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Button, ErrorText, Field } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import type { WaitlistSource } from '@/core/net/auth';
import { HttpError } from '@/core/net/http';
import { ensureSignedIn, getCloud, getLibrary } from '@/platform/services';

/** The email left on this device, so the form says "you are on the list" next time. */
const SETTING = 'waitlist_email';

/**
 * "Tell me when foolore is released" (ADR 66): an email for the release waitlist. Optional and
 * one field; the cloud keeps one row per email. `source` says where the player saw it.
 */
export function WaitlistForm({ source, intro }: { source: WaitlistSource; intro?: string }) {
  const [email, setEmail] = useState('');
  const [joined, setJoined] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    void getLibrary()
      .then((library) => library.repository.getSetting(SETTING))
      .then((saved) => active && setJoined(saved))
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  const join = async () => {
    const value = email.trim();
    if (!/^\S+@\S+\.\S+$/.test(value)) return setError('That does not look like an email address.');
    setBusy(true);
    setError('');
    try {
      await ensureSignedIn();
      await getCloud().auth.joinWaitlist(value, source);
      await (await getLibrary()).repository.setSetting(SETTING, value);
      setJoined(value);
    } catch (e) {
      setError(e instanceof HttpError && e.status === 422 ? 'That does not look like an email address.' : 'Could not reach the server. Try again in a moment.');
    } finally {
      setBusy(false);
    }
  };

  if (joined) {
    return (
      <ThemedText type="caption" themeColor="inkMuted">
        You are on the list: we will write to {joined} once foolore is released. Nothing else.
      </ThemedText>
    );
  }
  return (
    <View style={styles.form}>
      {intro ? <ThemedText type="body">{intro}</ThemedText> : null}
      <Field
        placeholder="you@example.com"
        value={email}
        onChangeText={setEmail}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="email-address"
        autoComplete="email"
        onSubmitEditing={() => void join()}
        returnKeyType="send"
      />
      <Button title="Notify me" variant="primary" busy={busy} disabled={!email.trim()} onPress={() => void join()} />
      <ErrorText>{error}</ErrorText>
      <ThemedText type="caption" themeColor="inkMuted">
        One email when the full version is out. No newsletters.
      </ThemedText>
    </View>
  );
}

const styles = StyleSheet.create({
  form: { gap: Spacing.two },
});
