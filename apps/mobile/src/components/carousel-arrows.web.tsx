import { Pressable, StyleSheet } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Spacing } from '@/constants/theme';

/** The previous or the next card with a mouse (stage T.3); the dots sit between the two. */
export function CarouselArrow({ next, at, count, onGo }: { next: boolean; at: number; count: number; onGo: (index: number) => void }) {
  const disabled = next ? at >= count - 1 : at === 0;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={next ? 'Next character' : 'Previous character'}
      disabled={disabled}
      onPress={() => onGo(next ? at + 1 : at - 1)}
      style={[styles.arrow, disabled && styles.off]}>
      <ThemedText type="title" themeColor="accent">
        {next ? '›' : '‹'}
      </ThemedText>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  arrow: { alignItems: 'center', justifyContent: 'center', minHeight: 44, minWidth: 44, paddingHorizontal: Spacing.two },
  off: { opacity: 0.3 },
});
