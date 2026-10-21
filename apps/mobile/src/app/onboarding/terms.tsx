import { router } from 'expo-router';
import { useState } from 'react';
import { StyleSheet } from 'react-native';

import { OnboardingPage } from '@/components/onboarding';
import { TERMS_POINTS, TermsSheet } from '@/components/terms';
import { ThemedText } from '@/components/themed-text';
import { Button, Card } from '@/components/ui';
import { acceptTerms } from '@/platform/onboarding';

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
      {TERMS_POINTS.map((point) => (
        <Card key={point.title}>
          <ThemedText type="label">{point.title}</ThemedText>
          {point.text ? (
            <ThemedText type="caption" themeColor="inkMuted">
              {point.text}
            </ThemedText>
          ) : null}
        </Card>
      ))}
      <TermsSheet visible={reading} onClose={() => setReading(false)} />
    </OnboardingPage>
  );
}

const styles = StyleSheet.create({
  draft: { letterSpacing: 1.5, fontWeight: 700 },
});
