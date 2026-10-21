/**
 * Prose with markup (ADR 67): the master's and the characters' text, and the player's lines.
 * Screens use only `<Prose>`; the parser behind it (`core/text/prose.ts`) can be swapped for a
 * library without touching them. Italic and bold are separate font faces for Literata (custom
 * fonts in SDK 57 have no weight mapping); the system sans takes `fontStyle` / `fontWeight`.
 */
import { useMemo } from 'react';
import { StyleSheet, Text, View, type StyleProp, type TextStyle } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Fonts, Spacing } from '@/constants/theme';
import { parseProse, type ProseSpan } from '@/core/text/prose';
import { useTheme } from '@/hooks/use-theme';

export interface ProseProps {
  text: string;
  /** `story` — Literata (the master, the characters); `body` — the system sans (the player). */
  type?: 'story' | 'body';
  /** Still streaming: an open marker styles the rest, a half-arrived one is hidden. */
  streaming?: boolean;
  style?: StyleProp<TextStyle>;
}

export function Prose({ text, type = 'story', streaming = false, style }: ProseProps) {
  const theme = useTheme();
  const blocks = useMemo(() => parseProse(text, { streaming }), [text, streaming]);
  const face = type === 'story' ? storyFace : sansFace;
  return (
    <View style={styles.blocks}>
      {blocks.map((block, i) => {
        const paragraph = (
          <ThemedText type={type} style={style}>
            {block.spans.map((span, j) => (
              <Text key={j} style={face(span)}>
                {span.text}
              </Text>
            ))}
          </ThemedText>
        );
        return block.kind === 'quote' ? (
          <View key={i} style={[styles.quote, { borderLeftColor: theme.gold }]}>
            {paragraph}
          </View>
        ) : (
          <View key={i}>{paragraph}</View>
        );
      })}
    </View>
  );
}

const storyFace = (span: ProseSpan): TextStyle | undefined =>
  span.bold ? { fontFamily: Fonts.storyBold } : span.italic ? { fontFamily: Fonts.storyItalic } : undefined;

const sansFace = (span: ProseSpan): TextStyle | undefined =>
  span.bold || span.italic ? { fontStyle: span.italic ? 'italic' : 'normal', fontWeight: span.bold ? 600 : 'normal' } : undefined;

const styles = StyleSheet.create({
  blocks: { gap: Spacing.three },
  quote: { borderLeftWidth: 2, paddingLeft: Spacing.three },
});
