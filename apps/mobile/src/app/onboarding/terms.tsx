import { router } from 'expo-router';
import { useState } from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';

import { OnboardingPage } from '@/components/onboarding';
import { ThemedText } from '@/components/themed-text';
import { Button, Card, Sheet } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { acceptTerms, TERMS_VERSION } from '@/platform/onboarding';

const POINTS = [
  { title: 'You must be 16 or older.', text: '' },
  { title: 'What you write goes to our servers', text: 'and to the AI providers that write the story with you.' },
  { title: 'The AI can be wrong.', text: 'It is fiction, not advice.' },
];

/** A stub (ADR 60): the real terms and privacy policy come from a lawyer (backlog). */
const FULL_TEXT = `DRAFT — not the final legal text.

foolore is a prototype of an interactive story with an AI game master.

Who can play. You must be at least 16 years old.

What we keep. The app keeps your stories, heroes and games on this device. To play, it sends what you write and the parts of the story the master needs to our servers, which pass them to AI model providers to write the answer. We count how much each guest plays to keep the prototype within its limit. If you leave an email, we keep it only to tell you when foolore is released.

The AI. The stories are written by AI models. They can be wrong, strange or unsuitable; they are fiction, not advice.

Your rights. You can delete the app and its data at any time. Ask us to delete your email: write to the address on the store page.

Version ${TERMS_VERSION}.`;

/** Terms (D.6): three points, the full text in a sheet, "Accept" (local, then the cloud). */
export default function TermsScreen() {
  const [reading, setReading] = useState(false);
  const [busy, setBusy] = useState(false);

  const accept = async () => {
    setBusy(true);
    try {
      await acceptTerms();
      router.push('/onboarding/hero');
    } finally {
      setBusy(false);
    }
  };

  return (
    <OnboardingPage
      step={0}
      footer={
        <>
          <Button title="Read the full terms" variant="ghost" onPress={() => setReading(true)} />
          <Button title="Accept and continue" variant="primary" busy={busy} onPress={() => void accept()} />
        </>
      }>
      <ThemedText type="title">Before we begin</ThemedText>
      <ThemedText type="caption" themeColor="danger" style={styles.draft}>
        DRAFT
      </ThemedText>
      {POINTS.map((point) => (
        <Card key={point.title}>
          <ThemedText type="label">{point.title}</ThemedText>
          {point.text ? (
            <ThemedText type="caption" themeColor="inkMuted">
              {point.text}
            </ThemedText>
          ) : null}
        </Card>
      ))}
      <Sheet visible={reading} onClose={() => setReading(false)} title="Terms and privacy">
        <ScrollView>
          <View style={styles.full}>
            <ThemedText type="body">{FULL_TEXT}</ThemedText>
          </View>
        </ScrollView>
        <Button title="Close" onPress={() => setReading(false)} />
      </Sheet>
    </OnboardingPage>
  );
}

const styles = StyleSheet.create({
  draft: { letterSpacing: 1.5, fontWeight: 700 },
  full: { paddingBottom: Spacing.three },
});
