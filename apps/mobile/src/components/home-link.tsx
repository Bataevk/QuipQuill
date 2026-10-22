import { router } from 'expo-router';
import { Pressable, StyleSheet } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Spacing } from '@/constants/theme';

/**
 * The header's way home when a page has nothing to go back to: opened by a link or reloaded in
 * the browser (ADR 69), or reached from the onboarding. Leads to the Stories tab.
 */
export function HomeLink() {
  return (
    <Pressable accessibilityRole="link" onPress={() => router.replace('/')} hitSlop={8} style={styles.link}>
      <ThemedText type="label" themeColor="accent">
        ‹ Stories
      </ThemedText>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  link: { paddingHorizontal: Spacing.two },
});
