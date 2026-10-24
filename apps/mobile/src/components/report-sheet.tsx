import { useState } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Button, ErrorText, Field, Sheet } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { HttpError } from '@/core/net/http';
import { ensureSignedIn } from '@/platform/services';

/**
 * The game header's "Report": the player says what went wrong; the game itself (hero, every line
 * they saw) goes with it, so the team can see the moment as it was.
 */
export function ReportButton({ send }: { send: (message: string) => Promise<void> }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Pressable accessibilityRole="button" onPress={() => setOpen(true)} hitSlop={8} style={styles.link}>
        <ThemedText type="label" themeColor="accent">
          Report
        </ThemedText>
      </Pressable>
      <ReportSheet visible={open} onClose={() => setOpen(false)} send={send} />
    </>
  );
}

function ReportSheet({ visible, onClose, send }: { visible: boolean; onClose: () => void; send: (message: string) => Promise<void> }) {
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState('');
  // The field keeps its own text (`Field`): a new key empties it after a report is sent.
  const [round, setRound] = useState(0);

  const close = () => {
    onClose();
    if (sent) {
      setSent(false);
      setMessage('');
      setRound((n) => n + 1);
    }
    setError('');
  };

  const submit = async () => {
    const text = message.trim();
    if (!text) return;
    setBusy(true);
    setError('');
    try {
      await ensureSignedIn();
      await send(text);
      setSent(true);
    } catch (e) {
      setError(e instanceof HttpError && e.status === 413 ? 'This game is too long to send whole. Tell us in a few words instead.' : 'Could not send the report. Try again in a moment.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <Sheet visible={visible} onClose={close} title={sent ? 'Thank you!' : 'Report a problem'}>
      {sent ? (
        <View style={styles.body}>
          <ThemedText type="body">We have your report and this game as it is now. It helps us make the master better.</ThemedText>
          <Button title="Back to the story" variant="primary" onPress={close} />
        </View>
      ) : (
        <View style={styles.body}>
          <ThemedText type="body">What went wrong? The master forgot something, broke a rule, or the app misbehaved — say it in your own words.</ThemedText>
          <Field
            key={round}
            placeholder="The master…"
            value={message}
            onChangeText={setMessage}
            multiline
            maxLength={4000}
            style={styles.field}
            autoFocus
          />
          <ThemedText type="caption" themeColor="inkMuted">
            Your hero and this game’s story so far are sent with it.
          </ThemedText>
          <ErrorText>{error}</ErrorText>
          <Button title="Send report" variant="primary" busy={busy} disabled={!message.trim()} onPress={() => void submit()} />
        </View>
      )}
    </Sheet>
  );
}

const styles = StyleSheet.create({
  link: { paddingHorizontal: Spacing.two },
  body: { gap: Spacing.two },
  field: { minHeight: 110, textAlignVertical: 'top' },
});
