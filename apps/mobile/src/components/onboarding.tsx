/** Pieces of the first launch (stage D.6): a parchment page with a back link and the step dots. */
import { router } from 'expo-router';
import type { ReactNode } from 'react';
import { KeyboardAvoidingView, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PaperBackground } from '@/components/paper';
import { Button, ProgressDots } from '@/components/ui';
import { MaxContentWidth, Spacing } from '@/constants/theme';

/** The steps after the welcome: terms, hero, release news, story. */
export const ONBOARDING_STEPS = 4;

/**
 * One onboarding page on paper (ADR 63): back and the dots on top, the content, the main action
 * at the bottom (`footer`). `onBack` replaces the default (the previous screen).
 */
export function OnboardingPage({
  step,
  children,
  footer,
  onBack,
  back = true,
}: {
  step?: number;
  children: ReactNode;
  footer?: ReactNode;
  onBack?: () => void;
  back?: boolean;
}) {
  return (
    <PaperBackground>
      <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
        {/* `padding` on Android too: with edge-to-edge the window does not shrink (stage T.1). */}
        <KeyboardAvoidingView style={styles.safe} behavior="padding">
          <View style={styles.bar}>
            <View style={styles.back}>{back ? <Button title="Back" variant="ghost" onPress={onBack ?? (() => router.back())} /> : null}</View>
            {step !== undefined ? <ProgressDots count={ONBOARDING_STEPS} index={step} /> : null}
            <View style={styles.back} />
          </View>
          <ScrollView contentContainerStyle={styles.body} keyboardShouldPersistTaps="handled">
            {children}
          </ScrollView>
          {footer ? <View style={styles.footer}>{footer}</View> : null}
        </KeyboardAvoidingView>
      </SafeAreaView>
    </PaperBackground>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, alignSelf: 'center', maxWidth: MaxContentWidth, width: '100%' },
  bar: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', minHeight: 48, paddingHorizontal: Spacing.two },
  back: { minWidth: 88 },
  body: { gap: Spacing.three, padding: Spacing.three, paddingTop: Spacing.two },
  footer: { gap: Spacing.two, paddingHorizontal: Spacing.three, paddingBottom: Spacing.three, paddingTop: Spacing.two },
});
