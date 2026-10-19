/** Building blocks of the screens, on the design system's tokens (stage D.1, ADR 63). */
import { useState, type ReactNode } from 'react';
import {
  ActivityIndicator,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  TextInput,
  type StyleProp,
  type TextInputProps,
  View,
  type ViewStyle,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PaperBackground } from '@/components/paper';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { MaxContentWidth, Radius, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

/**
 * A screen with a title, safe-area padding and (by default) scrolling content. `paper` puts it on
 * the parchment ground — only for screens outside the main pages (ADR 63).
 */
export function Screen({
  title,
  children,
  scroll = true,
  footer,
  paper = false,
}: {
  title?: string;
  children: ReactNode;
  scroll?: boolean;
  footer?: ReactNode;
  paper?: boolean;
}) {
  const body = (
    <View style={styles.body}>
      {title ? <ThemedText type="title">{title}</ThemedText> : null}
      {children}
    </View>
  );
  const content = (
    // The Stack header covers the top inset; only the bottom is ours.
    <SafeAreaView style={styles.safeArea} edges={['bottom']}>
      {scroll ? <ScrollView keyboardShouldPersistTaps="handled">{body}</ScrollView> : body}
      {footer}
    </SafeAreaView>
  );
  return paper ? (
    <PaperBackground style={styles.screen}>{content}</PaperBackground>
  ) : (
    <ThemedView style={styles.screen}>{content}</ThemedView>
  );
}

export type ButtonVariant = 'primary' | 'secondary' | 'ghost';

/** `primary` — the one main action of a screen; `secondary` — the rest; `ghost` — quiet links. */
export function Button({
  title,
  onPress,
  disabled,
  busy,
  selected,
  variant = 'secondary',
}: {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  busy?: boolean;
  selected?: boolean;
  variant?: ButtonVariant;
}) {
  const theme = useTheme();
  const off = disabled || busy;
  const look = {
    primary: { box: { backgroundColor: theme.accent, borderColor: theme.accent }, text: theme.onAccent },
    secondary: { box: { backgroundColor: theme.surface, borderColor: selected ? theme.accent : theme.line }, text: theme.ink },
    ghost: { box: { backgroundColor: 'transparent', borderColor: 'transparent' }, text: theme.accent },
  }[variant];
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled: !!off, selected: !!selected, busy: !!busy }}
      disabled={off}
      onPress={onPress}
      style={({ pressed }) => [
        styles.button,
        look.box,
        pressed && (variant === 'primary' ? styles.pressedPrimary : { backgroundColor: theme.sunken }),
        off && styles.disabled,
      ]}>
      {busy ? (
        <ActivityIndicator color={look.text} />
      ) : (
        <ThemedText type="label" style={{ color: look.text }}>
          {title}
        </ThemedText>
      )}
    </Pressable>
  );
}

/** A raised block: a story, a hero, a character proposal. */
export function Card({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) {
  const theme = useTheme();
  return <View style={[styles.card, { backgroundColor: theme.surface, borderColor: theme.line }, style]}>{children}</View>;
}

/**
 * A text field. Uncontrolled on purpose: `value` only seeds it, and `onChangeText` reports the
 * text. A controlled TextInput lets React write a stale value back while the player (or a
 * paste, or autocorrect) is still typing, and characters get lost. To clear it, remount it
 * with a new `key`.
 */
export function Field({ label, value, onFocus, onBlur, ...props }: TextInputProps & { label?: string }) {
  const theme = useTheme();
  const [focused, setFocused] = useState(false);
  return (
    <View style={styles.field}>
      {label ? (
        <ThemedText type="caption" themeColor="inkMuted">
          {label}
        </ThemedText>
      ) : null}
      <TextInput
        placeholderTextColor={theme.inkMuted}
        selectionColor={theme.accent}
        cursorColor={theme.accent}
        defaultValue={value}
        onFocus={(e) => {
          setFocused(true);
          onFocus?.(e);
        }}
        onBlur={(e) => {
          setFocused(false);
          onBlur?.(e);
        }}
        {...props}
        style={[
          styles.input,
          { color: theme.ink, backgroundColor: theme.surface, borderColor: focused ? theme.accent : theme.line },
          props.style,
        ]}
      />
    </View>
  );
}

/** One choice among a few short ones (sex, origin, theme). */
export function Chip({ label, selected, onPress }: { label: string; selected?: boolean; onPress?: () => void }) {
  const theme = useTheme();
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ selected: !!selected }}
      disabled={!onPress}
      onPress={onPress}
      style={({ pressed }) => [
        styles.chip,
        {
          backgroundColor: pressed ? theme.sunken : theme.surface,
          borderColor: selected ? theme.accent : theme.line,
        },
      ]}>
      <ThemedText type="caption" themeColor={selected ? 'accent' : 'ink'} style={selected && styles.chipSelected}>
        {label}
      </ThemedText>
    </Pressable>
  );
}

/** Where the player is in a short sequence (onboarding, the hero's steps). */
export function ProgressDots({ count, index }: { count: number; index: number }) {
  const theme = useTheme();
  return (
    <View
      style={styles.dots}
      accessible
      accessibilityRole="progressbar"
      accessibilityValue={{ min: 1, max: count, now: index + 1 }}>
      {Array.from({ length: count }, (_, i) => (
        <View key={i} style={[styles.dot, { backgroundColor: i <= index ? theme.accent : theme.line }]} />
      ))}
    </View>
  );
}

/** A panel from the bottom over a dimmed screen; a tap on the dim or the system back closes it. */
export function Sheet({
  visible,
  onClose,
  title,
  children,
}: {
  visible: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
}) {
  const theme = useTheme();
  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <View style={styles.sheetRoot}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Close"
          style={[StyleSheet.absoluteFill, { backgroundColor: theme.scrim }]}
          onPress={onClose}
        />
        <SafeAreaView edges={['bottom']} style={[styles.sheet, { backgroundColor: theme.surface, borderColor: theme.line }]}>
          <View style={[styles.grab, { backgroundColor: theme.line }]} />
          {title ? <ThemedText type="title">{title}</ThemedText> : null}
          {children}
        </SafeAreaView>
      </View>
    </Modal>
  );
}

export function ErrorText({ children }: { children?: ReactNode }) {
  if (!children) return null;
  return (
    <ThemedText type="caption" themeColor="danger" accessibilityRole="alert">
      {children}
    </ThemedText>
  );
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
      accessibilityState={{ selected: !!selected }}
      disabled={!onPress}
      onPress={onPress}
      style={({ pressed }) => [
        styles.card,
        styles.row,
        { backgroundColor: pressed ? theme.sunken : theme.surface, borderColor: selected ? theme.accent : theme.line },
      ]}>
      <ThemedText type="label">{title}</ThemedText>
      {subtitle ? (
        <ThemedText type="caption" themeColor="inkMuted">
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
    borderRadius: Radius.medium,
    borderWidth: StyleSheet.hairlineWidth * 2,
    justifyContent: 'center',
    minHeight: 48,
    paddingHorizontal: Spacing.three,
    paddingVertical: Spacing.two,
  },
  pressedPrimary: { opacity: 0.85 },
  disabled: { opacity: 0.45 },
  card: { borderRadius: Radius.medium, borderWidth: StyleSheet.hairlineWidth * 2, gap: Spacing.two, padding: Spacing.three },
  row: { gap: Spacing.half },
  field: { gap: Spacing.one },
  input: {
    borderRadius: Radius.medium,
    borderWidth: StyleSheet.hairlineWidth * 2,
    fontSize: 16,
    minHeight: 48,
    paddingHorizontal: Spacing.three,
    paddingVertical: Spacing.two + Spacing.one,
  },
  chip: {
    borderRadius: Radius.pill,
    borderWidth: StyleSheet.hairlineWidth * 2,
    minHeight: 36,
    justifyContent: 'center',
    paddingHorizontal: Spacing.three,
  },
  chipSelected: { fontWeight: 600 },
  dots: { flexDirection: 'row', gap: Spacing.two - Spacing.half },
  dot: { borderRadius: Radius.pill, height: 6, width: 6 },
  sheetRoot: { flex: 1, justifyContent: 'flex-end' },
  sheet: {
    borderTopLeftRadius: Radius.large,
    borderTopRightRadius: Radius.large,
    borderWidth: StyleSheet.hairlineWidth * 2,
    borderBottomWidth: 0,
    gap: Spacing.three,
    maxHeight: '85%',
    paddingHorizontal: Spacing.three,
    paddingTop: Spacing.two,
    paddingBottom: Spacing.three,
  },
  grab: { alignSelf: 'center', borderRadius: Radius.pill, height: 4, width: 36 },
});
