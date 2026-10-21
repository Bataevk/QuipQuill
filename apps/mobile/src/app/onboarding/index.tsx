import { router } from 'expo-router';
import { StyleSheet, View } from 'react-native';

import { FoolEmblem } from '@/components/fool-loader';
import { OnboardingPage } from '@/components/onboarding';
import { ThemedText } from '@/components/themed-text';
import { Button } from '@/components/ui';
import { Spacing } from '@/constants/theme';

/** Welcome (D.6): the emblem, one line, one button. No sign-in here (ADR 66). */
export default function WelcomeScreen() {
  return (
    <OnboardingPage back={false} footer={<Button title="Begin" variant="primary" onPress={() => router.push('/onboarding/terms')} />}>
      <View style={styles.hero}>
        <FoolEmblem size={180} />
        <ThemedText type="display" style={styles.center}>
          foolore
        </ThemedText>
        <ThemedText type="story" themeColor="inkMuted" style={styles.center}>
          Stories that answer back. You say what your hero does — the world and its people respond.
        </ThemedText>
      </View>
    </OnboardingPage>
  );
}

const styles = StyleSheet.create({
  hero: { alignItems: 'center', gap: Spacing.three, paddingTop: Spacing.six },
  center: { textAlign: 'center' },
});
