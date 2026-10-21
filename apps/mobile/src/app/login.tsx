import { router } from 'expo-router';
import { useState } from 'react';

import { ThemedText } from '@/components/themed-text';
import { Button, ErrorText, Field, Screen, errorText } from '@/components/ui';
import { HttpError } from '@/core/net/http';
import { getCloud } from '@/platform/services';

/** What the cloud's auth errors mean for the player (cloud/README.md#авторизация). */
function loginError(e: unknown): string {
  if (!(e instanceof HttpError)) return errorText(e);
  if (e.status === 401) return 'Wrong email or password.';
  if (e.status === 403) return 'This invite code is not valid.';
  if (e.status === 409) return 'This email is already registered — sign in instead.';
  if (e.status === 422) return 'Check the email address and the password.';
  return errorText(e);
}

/**
 * Email + password; an invite code registers a new account (ADR 17). Opened from the profile: the
 * prototype plays as a guest (ADR 66), this is for the team's accounts. It replaces the guest.
 */
export default function LoginScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [invite, setInvite] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const submit = async () => {
    setBusy(true);
    setError('');
    try {
      const { auth, socket } = getCloud();
      if (invite.trim()) await auth.register(email.trim(), password, invite.trim());
      else await auth.login(email.trim(), password);
      // The socket signs in again with the new account.
      socket.stop();
      socket.start();
      router.back();
    } catch (e) {
      setError(loginError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Screen title="Sign in" paper>
      <Field label="Email" value={email} onChangeText={setEmail} autoCapitalize="none" keyboardType="email-address" autoComplete="email" />
      <Field label="Password" value={password} onChangeText={setPassword} secureTextEntry autoComplete="password" />
      <Field label="Invite code (new account only)" value={invite} onChangeText={setInvite} autoCapitalize="none" />
      <Button title={invite.trim() ? 'Create account' : 'Sign in'} variant="primary" onPress={submit} busy={busy} disabled={!email || !password} />
      <ErrorText>{error}</ErrorText>
      <ThemedText type="caption" themeColor="inkMuted">
        For the team and invited testers. Without an account you play as a guest of this device.
      </ThemedText>
    </Screen>
  );
}
