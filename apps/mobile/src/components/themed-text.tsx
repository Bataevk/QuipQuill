import { StyleSheet, Text, type TextProps } from 'react-native';

import { Fonts, ThemeColor } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

/**
 * The type scale (ADR 63): `display` and `title` in IM Fell English, `story` in Literata
 * (the master's and the characters' text), the rest in the system sans.
 */
export type TextType = 'display' | 'title' | 'story' | 'body' | 'label' | 'caption' | 'mono';

export type ThemedTextProps = TextProps & {
  type?: TextType;
  themeColor?: ThemeColor;
};

export function ThemedText({ style, type = 'body', themeColor, ...rest }: ThemedTextProps) {
  const theme = useTheme();
  return <Text style={[{ color: theme[themeColor ?? 'ink'] }, styles[type], style]} {...rest} />;
}

const styles = StyleSheet.create({
  display: { fontFamily: Fonts.display, fontSize: 40, lineHeight: 46 },
  title: { fontFamily: Fonts.display, fontSize: 28, lineHeight: 34 },
  story: { fontFamily: Fonts.story, fontSize: 17, lineHeight: 26 },
  body: { fontFamily: Fonts.sans, fontSize: 16, lineHeight: 22 },
  label: { fontFamily: Fonts.sans, fontSize: 15, lineHeight: 20, fontWeight: 600 },
  caption: { fontFamily: Fonts.sans, fontSize: 13, lineHeight: 18 },
  mono: { fontFamily: Fonts.mono, fontSize: 12, lineHeight: 16 },
});
