import type { ReactNode } from 'react';
import { Image, StyleSheet, type StyleProp, View, type ViewStyle } from 'react-native';

import { useScheme, useTheme } from '@/hooks/use-theme';

// Rendered by icons/render.mjs (ADR 63): alpha-only specks over the theme's bg. The grain is a
// small tile, repeated; the mottling is one low-frequency image stretched over the screen, so
// it has no seams. Together they bring the mean to the splash's colour.
const GRAIN = {
  light: require('@/assets/images/paper-grain-light.png'),
  dark: require('@/assets/images/paper-grain-dark.png'),
};
const MOTTLE = {
  light: require('@/assets/images/paper-mottle-light.png'),
  dark: require('@/assets/images/paper-mottle-dark.png'),
};

/**
 * The parchment ground (ADR 63): splash-adjacent screens, loading, onboarding, covers.
 * The main pages (tabs, heroes, stories, profile) and the chat stay on the flat `bg`.
 */
export function PaperBackground({ children, style }: { children?: ReactNode; style?: StyleProp<ViewStyle> }) {
  const scheme = useScheme();
  const theme = useTheme();
  return (
    <View style={[styles.fill, { backgroundColor: theme.bg }, style]}>
      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        <Image source={MOTTLE[scheme]} resizeMode="stretch" style={styles.layer} accessible={false} />
        <Image source={GRAIN[scheme]} resizeMode="repeat" style={styles.layer} accessible={false} />
      </View>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  // A static image keeps its own size unless told otherwise, even when absolutely positioned.
  layer: { position: 'absolute', left: 0, top: 0, height: '100%', width: '100%' },
});
