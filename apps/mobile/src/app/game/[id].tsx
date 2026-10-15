import { useLocalSearchParams } from 'expo-router';
import { useRef, useState } from 'react';
import { ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Button, ErrorText, Field } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import type { CharacterOption } from '@/core/agent/character';
import { PROPOSAL_CARD } from '@/core/agent/session';
import { useGame } from '@/hooks/use-game';
import { useTheme } from '@/hooks/use-theme';

/** The game (stage 5.2): bubbles of the master, NPCs and the player, live streaming, cards. */
export default function GameScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const game = useGame(id);
  const [text, setText] = useState('');
  const [inputKey, setInputKey] = useState(0); // a new key clears the (uncontrolled) field
  const scroll = useRef<ScrollView>(null);

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
  const lastCard = game.lines.findLastIndex((l) => l.extra.kind === PROPOSAL_CARD);

  return (
    <ThemedView style={styles.screen}>
      <SafeAreaView style={styles.screen} edges={['bottom']}>
        <KeyboardAvoidingView
          style={styles.screen}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          keyboardVerticalOffset={Platform.OS === 'ios' ? 100 : 0}>
          <ScrollView
            ref={scroll}
            contentContainerStyle={styles.feed}
            onContentSizeChange={() => scroll.current?.scrollToEnd({ animated: true })}
            keyboardShouldPersistTaps="handled">
            {game.lines.map((line, i) => (
              <View key={line.seq}>
                <Bubble speaker={line.speaker} name={line.extra.name as string | undefined} text={line.text} />
                {i === lastCard && adapting && game.proposal ? (
                  <Cards options={game.proposal.options} disabled={game.busy} onChoose={game.choose} />
                ) : null}
              </View>
            ))}
            {game.pending ? <Bubble speaker="player" text={game.pending} /> : null}
            {game.live.map((bubble) => (
              <Bubble key={bubble.requestId} speaker={bubble.channel} name={bubble.speaker} text={bubble.text} />
            ))}
            {game.busy && (game.thinking || game.live.length === 0) ? <ActivityIndicator style={styles.spinner} /> : null}
            {!game.online ? (
              <ThemedText type="small" themeColor="textSecondary">
                Connecting to the server… {game.busy ? 'Your move will be sent as soon as it is back.' : ''}
              </ThemedText>
            ) : null}
            <ErrorText>{game.error}</ErrorText>
            {(game.error || game.unfinished) && !game.busy ? (
              <Button title={game.error ? 'Try again' : 'The story paused — continue'} onPress={game.recover} />
            ) : null}
          </ScrollView>
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
              <Button title="Send" onPress={submit} disabled={!canType || !text.trim()} />
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
  const npc = speaker.startsWith('npc:');
  return (
    <View
      style={[
        styles.bubble,
        player ? styles.player : styles.other,
        { backgroundColor: player ? theme.backgroundSelected : theme.backgroundElement },
      ]}>
      {npc ? <ThemedText type="smallBold">{name ?? 'Someone'}</ThemedText> : null}
      <ThemedText>{text}</ThemedText>
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
        <View key={i} style={styles.card}>
          <ThemedText type="smallBold">
            {option.name} — {[option.race, option.age ? `${option.age}` : ''].filter(Boolean).join(', ')}
          </ThemedText>
          <ThemedText type="small">{option.description}</ThemedText>
          <Button title={i === 0 ? 'Play this character' : 'Play this one instead'} disabled={disabled} onPress={() => onChoose(i)} />
        </View>
      ))}
    </View>
  );
}


const styles = StyleSheet.create({
  screen: { flex: 1 },
  feed: { gap: Spacing.two, padding: Spacing.three },
  bubble: { borderRadius: Spacing.three, gap: Spacing.one, maxWidth: '88%', padding: Spacing.three },
  player: { alignSelf: 'flex-end' },
  other: { alignSelf: 'flex-start' },
  spinner: { alignSelf: 'flex-start', padding: Spacing.two },
  cards: { gap: Spacing.two, marginTop: Spacing.two },
  card: { gap: Spacing.two },
  inputRow: { alignItems: 'center', flexDirection: 'row', gap: Spacing.two, padding: Spacing.two },
  input: { flex: 1 },
});
