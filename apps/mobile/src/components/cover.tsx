import { Pressable, StyleSheet, View } from 'react-native';

import { PaperBackground } from '@/components/paper';
import { ThemedText } from '@/components/themed-text';
import { Radius, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

/**
 * A typographic story cover (ADR 63): parchment, a thin gold frame, the title in IM Fell.
 * Drawn covers come later (backlog); this one costs nothing and stays on brand.
 */
export function Cover({
  title,
  subtitle,
  large = false,
  onPress,
}: {
  title: string;
  subtitle?: string;
  large?: boolean;
  onPress?: () => void;
}) {
  const theme = useTheme();
  const body = (
    <PaperBackground style={[styles.paper, large ? styles.large : styles.small, { borderColor: theme.line }]}>
      <View pointerEvents="none" style={[styles.frame, { borderColor: theme.gold }]} />
      <ThemedText type={large ? 'display' : 'title'} style={styles.center} numberOfLines={3}>
        {title}
      </ThemedText>
      {subtitle ? (
        <ThemedText type="caption" themeColor="inkMuted" style={styles.center} numberOfLines={2}>
          {subtitle}
        </ThemedText>
      ) : null}
    </PaperBackground>
  );
  if (!onPress) return body;
  return (
    <Pressable accessibilityRole="button" accessibilityLabel={title} onPress={onPress} style={({ pressed }) => pressed && styles.pressed}>
      {body}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  paper: {
    alignItems: 'center',
    borderRadius: Radius.medium,
    borderWidth: StyleSheet.hairlineWidth * 2,
    flex: 0,
    gap: Spacing.two,
    justifyContent: 'center',
    overflow: 'hidden',
    paddingHorizontal: Spacing.four,
  },
  small: { minHeight: 150, paddingVertical: Spacing.four },
  large: { minHeight: 220, paddingVertical: Spacing.five },
  frame: {
    borderRadius: Radius.small,
    borderWidth: 1,
    bottom: Spacing.two,
    left: Spacing.two,
    opacity: 0.8,
    position: 'absolute',
    right: Spacing.two,
    top: Spacing.two,
  },
  center: { textAlign: 'center' },
  pressed: { opacity: 0.85 },
});
