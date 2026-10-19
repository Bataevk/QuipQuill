import { useLocalSearchParams } from 'expo-router';
import { useMemo, useRef, useState } from 'react';
import { ActivityIndicator, FlatList, KeyboardAvoidingView, Platform, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Button, Card, ErrorText, Field } from '@/components/ui';
import { Radius, Spacing } from '@/constants/theme';
import type { CharacterOption } from '@/core/agent/character';
import { PROPOSAL_CARD } from '@/core/agent/session';
import { isActorChannel } from '@/core/protocol/channels';
import { useGame } from '@/hooks/use-game';
import { useTheme } from '@/hooks/use-theme';

/** One row of the feed: a transcript line (maybe with the character cards), the pending move, a live bubble. */
type FeedItem =
  | { key: string; kind: 'bubble'; speaker: string; name?: string; text: string; cards?: CharacterOption[] }
  | { key: string; kind: 'cost'; text: string }
  | { key: string; kind: 'footer' };

/** The game (stage 5.2): bubbles of the master, NPCs and the player, live streaming, cards. */
export default function GameScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const game = useGame(id);
  const [text, setText] = useState('');
  const [inputKey, setInputKey] = useState(0); // a new key clears the (uncontrolled) field
  const list = useRef<FlatList<FeedItem>>(null);
  const theme = useTheme();

  const adapting = game.phase === 'adapting';
  const canType = game.ready && !game.busy && (game.phase === 'playing' || (adapting && !!game.proposal?.canClarify));
  const submit = () => {
    const value = text.trim();
    if (!value) return;
    setText('');
    setInputKey((k) => k + 1);
    if (adapting) game.clarify(value);
    else game.send(value);
  };

  const items = useMemo<FeedItem[]>(() => {
    const lastCard = game.lines.findLastIndex((l) => l.extra.kind === PROPOSAL_CARD);
    const rows: FeedItem[] = [];
    game.lines.forEach((line, i) => {
      rows.push({
        key: `line-${line.seq}`,
        kind: 'bubble',
        speaker: line.speaker,
        name: line.extra.name as string | undefined,
        text: line.text,
        cards: i === lastCard && adapting ? game.proposal?.options : undefined,
      });
      game.costs
        .filter((cost) => cost.afterSeq === line.seq)
        .forEach((cost, n) => rows.push({ key: `cost-${line.seq}-${n}`, kind: 'cost', text: cost.text }));
    });
    if (game.pending) rows.push({ key: 'pending', kind: 'bubble', speaker: 'player', text: game.pending });
    for (const bubble of game.live) {
      rows.push({ key: `live-${bubble.requestId}`, kind: 'bubble', speaker: bubble.channel, name: bubble.speaker, text: bubble.text });
    }
    rows.push({ key: 'footer', kind: 'footer' });
    return rows;
  }, [game.lines, game.pending, game.live, game.proposal, game.costs, adapting]);

  const footer = (
    <View style={styles.footer}>
      {game.busy && (game.thinking || game.live.length === 0) ? <ActivityIndicator color={theme.accent} style={styles.spinner} /> : null}
      {!game.online ? (
        <ThemedText type="caption" themeColor="inkMuted">
          Connecting to the server… {game.busy ? 'Your move will be sent as soon as it is back.' : ''}
        </ThemedText>
      ) : null}
      <ErrorText>{game.error}</ErrorText>
      {game.sessionCost ? (
        <ThemedText type="mono" themeColor="inkMuted">
          Session: {game.sessionCost}
        </ThemedText>
      ) : null}
      {(game.error || game.stalled) && !game.busy ? (
        <Button title={game.error ? 'Try again' : 'The story paused — continue'} onPress={game.recover} />
      ) : null}
    </View>
  );

  return (
    <ThemedView style={styles.screen}>
      <SafeAreaView style={styles.screen} edges={['bottom']}>
        <KeyboardAvoidingView
          style={styles.screen}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          keyboardVerticalOffset={Platform.OS === 'ios' ? 100 : 0}>
          <FlatList
            ref={list}
            data={items}
            keyExtractor={(item) => item.key}
            contentContainerStyle={styles.feed}
            keyboardShouldPersistTaps="handled"
            onContentSizeChange={() => list.current?.scrollToEnd({ animated: true })}
            renderItem={({ item }) =>
              item.kind === 'footer' ? (
                footer
              ) : item.kind === 'cost' ? (
                <ThemedText type="mono" themeColor="inkMuted">
                  {item.text}
                </ThemedText>
              ) : (
                <View>
                  <Bubble speaker={item.speaker} name={item.name} text={item.text} />
                  {item.cards ? <Cards options={item.cards} disabled={game.busy} onChoose={game.choose} /> : null}
                </View>
              )
            }
          />
          {game.phase === 'playing' || adapting ? (
            <View style={styles.inputRow}>
              <View style={styles.input}>
                <Field
                  key={inputKey}
                  onChangeText={setText}
                  editable={canType}
                  placeholder={adapting ? (game.proposal?.canClarify ? 'Ask for changes…' : 'Choose a card') : 'What do you do?'}
                  onSubmitEditing={submit}
                  returnKeyType="send"
                />
              </View>
              <Button title="Send" variant="primary" onPress={submit} disabled={!canType || !text.trim()} />
            </View>
          ) : null}
        </KeyboardAvoidingView>
      </SafeAreaView>
    </ThemedView>
  );
}

function Bubble({ speaker, name, text }: { speaker: string; name?: string; text: string }) {
  const theme = useTheme();
  const player = speaker === 'player';
  const npc = isActorChannel(speaker);
  return (
    <View
      style={[
        styles.bubble,
        player ? styles.player : styles.other,
        player
          ? { backgroundColor: theme.sunken, borderColor: theme.sunken }
          : { backgroundColor: theme.surface, borderColor: theme.line },
      ]}>
      {npc ? (
        <ThemedText type="label" themeColor="accent">
          {name ?? 'Someone'}
        </ThemedText>
      ) : null}
      <ThemedText type={player ? 'body' : 'story'}>{text}</ThemedText>
    </View>
  );
}

function Cards({
  options,
  disabled,
  onChoose,
}: {
  options: CharacterOption[];
  disabled: boolean;
  onChoose: (index: number) => void;
}) {
  return (
    <View style={styles.cards}>
      {options.map((option, i) => (
        <Card key={i}>
          <ThemedText type="title">{option.name}</ThemedText>
          <ThemedText type="caption" themeColor="inkMuted">
            {[option.race, option.age ? `${option.age}` : ''].filter(Boolean).join(', ')}
          </ThemedText>
          <ThemedText type="story">{option.description}</ThemedText>
          <Button
            title={i === 0 ? 'Play this character' : 'Play this one instead'}
            variant={i === 0 ? 'primary' : 'secondary'}
            disabled={disabled}
            onPress={() => onChoose(i)}
          />
        </Card>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  feed: { gap: Spacing.two, padding: Spacing.three },
  footer: { gap: Spacing.two },
  bubble: { borderRadius: Radius.large, borderWidth: StyleSheet.hairlineWidth * 2, gap: Spacing.one, maxWidth: '88%', padding: Spacing.three },
  player: { alignSelf: 'flex-end' },
  other: { alignSelf: 'flex-start' },
  spinner: { alignSelf: 'flex-start', padding: Spacing.two },
  cards: { gap: Spacing.two, marginTop: Spacing.two },
  inputRow: { alignItems: 'center', flexDirection: 'row', gap: Spacing.two, padding: Spacing.two },
  input: { flex: 1 },
});
