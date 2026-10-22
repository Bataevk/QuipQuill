import { useRef, useState } from 'react';
import { type NativeScrollEvent, type NativeSyntheticEvent, ScrollView, StyleSheet, View } from 'react-native';

import { CarouselArrow } from '@/components/carousel-arrows';
import type { ProposalCards } from '@/components/feed-rows';
import { Prose } from '@/components/prose';
import { ThemedText } from '@/components/themed-text';
import { Button, Card } from '@/components/ui';
import { Radius, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

/** A card takes most of the width, so the edge of the next one shows: there is more to swipe. */
const SHARE = 0.85;
const MAX_CARD = 340;

/**
 * The character cards of one proposal side by side, swiped one by one (stage T.3). Until the hero
 * is chosen every proposal stays open — an earlier one too; then the picked card is marked and the
 * rest are dimmed. On the web, arrows as well (`CarouselArrow`): a mouse cannot swipe.
 */
export function CardCarousel({
  cards,
  disabled,
  onChoose,
}: {
  cards: ProposalCards;
  disabled: boolean;
  onChoose: (index: number, round: number) => void;
}) {
  const theme = useTheme();
  const scroll = useRef<ScrollView>(null);
  const [width, setWidth] = useState(0);
  const [at, setAt] = useState(0);
  const cardWidth = Math.min(Math.round(width * SHARE), MAX_CARD);
  const step = cardWidth + Spacing.two;
  const count = cards.options.length;
  const go = (index: number) => {
    const next = Math.max(0, Math.min(count - 1, index));
    scroll.current?.scrollTo({ x: next * step, animated: true });
    setAt(next);
  };
  const onScrolled = (e: NativeSyntheticEvent<NativeScrollEvent>) => {
    if (step > 0) setAt(cardAt(e.nativeEvent, step, count));
  };
  return (
    <View style={styles.box} onLayout={(e) => setWidth(e.nativeEvent.layout.width)}>
      {cards.open && !cards.latest ? (
        <ThemedText type="caption" themeColor="inkMuted">
          An earlier proposal — you can still choose from it.
        </ThemedText>
      ) : null}
      {width > 0 ? (
        <ScrollView
          ref={scroll}
          horizontal
          showsHorizontalScrollIndicator={false}
          snapToInterval={step}
          decelerationRate="fast"
          disableIntervalMomentum
          onMomentumScrollEnd={onScrolled}
          onScroll={onScrolled}
          scrollEventThrottle={64}
          contentContainerStyle={styles.row}>
          {cards.options.map((option, i) => {
            const chosen = cards.chosen === i;
            const dimmed = !cards.open && !chosen;
            return (
              <Card
                key={i}
                style={[
                  { width: cardWidth },
                  chosen && { borderColor: theme.accent },
                  dimmed && styles.dimmed,
                ]}>
                {chosen ? (
                  <ThemedText type="label" themeColor="accent">
                    Chosen
                  </ThemedText>
                ) : null}
                <ThemedText type="title">{option.name}</ThemedText>
                <ThemedText type="caption" themeColor="inkMuted">
                  {[option.race, option.age ? `${option.age}` : ''].filter(Boolean).join(', ')}
                </ThemedText>
                <Prose text={option.description} />
                {cards.open ? (
                  <Button
                    title={i === 0 ? 'Play this character' : 'Play this one instead'}
                    variant={i === 0 && cards.latest ? 'primary' : 'secondary'}
                    disabled={disabled}
                    onPress={() => onChoose(i, cards.round)}
                  />
                ) : null}
              </Card>
            );
          })}
        </ScrollView>
      ) : null}
      {count > 1 ? (
        <View style={styles.under}>
          <CarouselArrow next={false} at={at} count={count} onGo={go} />
          <View style={styles.dots} accessibilityLabel={`Character ${at + 1} of ${count}`}>
            {cards.options.map((_, i) => (
              <View key={i} style={[styles.dot, { backgroundColor: i === at ? theme.accent : theme.line }]} />
            ))}
          </View>
          <CarouselArrow next at={at} count={count} onGo={go} />
        </View>
      ) : null}
    </View>
  );
}

/**
 * The card in view: by snap steps, but the end of the row is the last card — on a wide screen the
 * row stops scrolling before the last card's own step.
 */
function cardAt({ contentOffset, contentSize, layoutMeasurement }: NativeScrollEvent, step: number, count: number): number {
  if (contentOffset.x >= contentSize.width - layoutMeasurement.width - 2) return count - 1;
  return Math.max(0, Math.min(count - 1, Math.round(contentOffset.x / step)));
}

const styles = StyleSheet.create({
  box: { gap: Spacing.two, marginTop: Spacing.two },
  row: { gap: Spacing.two, paddingRight: Spacing.two },
  dimmed: { opacity: 0.55 },
  under: { alignItems: 'center', flexDirection: 'row', justifyContent: 'center', gap: Spacing.three, minHeight: 24 },
  dots: { flexDirection: 'row', gap: Spacing.two - Spacing.half },
  dot: { borderRadius: Radius.pill, height: 6, width: 6 },
});
