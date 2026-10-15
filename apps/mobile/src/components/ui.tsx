/** Plain building blocks for the prototype screens (stage 5.1; no design yet). */
import type { ReactNode } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  TextInput,
  type TextInputProps,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { MaxContentWidth, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

/** A screen with a title, safe-area padding and (by default) scrolling content. */
export function Screen({
  title,
  children,
  scroll = true,
  footer,
}: {
  title?: string;
  children: ReactNode;
  scroll?: boolean;
  footer?: ReactNode;
}) {
  const body = (
    <View style={styles.body}>
      {title ? <ThemedText type="subtitle">{title}</ThemedText> : null}
      {children}
    </View>
  );
  return (
    <ThemedView style={styles.screen}>
      <SafeAreaView style={styles.safeArea} edges={['top', 'bottom']}>
        {scroll ? <ScrollView keyboardShouldPersistTaps="handled">{body}</ScrollView> : body}
        {footer}
      </SafeAreaView>
    </ThemedView>
  );
}

export function Button({
  title,
  onPress,
  disabled,
  busy,
  selected,
}: {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  busy?: boolean;
  selected?: boolean;
}) {
  const theme = useTheme();
  const off = disabled || busy;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled: !!off, selected: !!selected }}
      disabled={off}
      onPress={onPress}
      style={[
        styles.button,
        { backgroundColor: selected ? theme.backgroundSelected : theme.backgroundElement },
        off && styles.disabled,
      ]}>
      {busy ? <ActivityIndicator /> : <ThemedText type="smallBold">{title}</ThemedText>}
    </Pressable>
  );
}

/**
 * A text field. Uncontrolled on purpose: `value` only seeds it, and `onChangeText` reports the
 * text. A controlled TextInput lets React write a stale value back while the player (or a
 * paste, or autocorrect) is still typing, and characters get lost. To clear it, remount it
 * with a new `key`.
 */
export function Field({ label, value, ...props }: TextInputProps & { label?: string }) {
  const theme = useTheme();
  return (
    <View style={styles.field}>
      {label ? (
        <ThemedText type="small" themeColor="textSecondary">
          {label}
        </ThemedText>
      ) : null}
      <TextInput
        placeholderTextColor={theme.textSecondary}
        defaultValue={value}
        {...props}
        style={[styles.input, { color: theme.text, backgroundColor: theme.backgroundElement }, props.style]}
      />
    </View>
  );
}

export function ErrorText({ children }: { children?: ReactNode }) {
  if (!children) return null;
  return <ThemedText style={styles.error}>{children}</ThemedText>;
}

/** A tappable card in a list (stories, characters, saves). */
export function Row({
  title,
  subtitle,
  onPress,
  selected,
}: {
  title: string;
  subtitle?: string;
  onPress?: () => void;
  selected?: boolean;
}) {
  const theme = useTheme();
  return (
    <Pressable
      accessibilityRole={onPress ? 'button' : undefined}
      disabled={!onPress}
      onPress={onPress}
      style={[styles.row, { backgroundColor: selected ? theme.backgroundSelected : theme.backgroundElement }]}>
      <ThemedText type="smallBold">{title}</ThemedText>
      {subtitle ? (
        <ThemedText type="small" themeColor="textSecondary">
          {subtitle}
        </ThemedText>
      ) : null}
    </Pressable>
  );
}

export const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

const styles = StyleSheet.create({
  screen: { flex: 1, flexDirection: 'row', justifyContent: 'center' },
  safeArea: { flex: 1, maxWidth: MaxContentWidth },
  body: { gap: Spacing.three, paddingHorizontal: Spacing.three, paddingVertical: Spacing.three },
  button: {
    alignItems: 'center',
    borderRadius: Spacing.two,
    paddingHorizontal: Spacing.three,
    paddingVertical: Spacing.two + Spacing.one,
  },
  disabled: { opacity: 0.5 },
  field: { gap: Spacing.one },
  input: { borderRadius: Spacing.two, fontSize: 16, paddingHorizontal: Spacing.three, paddingVertical: Spacing.two },
  error: { color: '#d93025' },
  row: { borderRadius: Spacing.two, gap: Spacing.half, padding: Spacing.three },
});
