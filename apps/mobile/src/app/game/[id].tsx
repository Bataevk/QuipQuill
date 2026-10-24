import { Stack, useLocalSearchParams } from 'expo-router';
import { useHeaderHeight } from 'expo-router/react-navigation';
import { useEffect, useMemo, useRef, useState } from 'react';
import { FlatList, KeyboardAvoidingView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { CardCarousel } from '@/components/card-carousel';
import { ChatInput } from '@/components/chat-input';
import { feedRows, type FeedRow } from '@/components/feed-rows';
import { FoolBells } from '@/components/fool-loader';
import { Prose } from '@/components/prose';
import { ReportButton } from '@/components/report-sheet';
import { WaitlistForm } from '@/components/waitlist-form';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Button, Card, ErrorText } from '@/components/ui';
import { MaxContentWidth, Radius, Spacing } from '@/constants/theme';
import { isActorChannel } from '@/core/protocol/channels';
import { useGame } from '@/hooks/use-game';
import { useTheme } from '@/hooks/use-theme';
import { ensureSignedIn, getCloud, getLibrary } from '@/platform/services';

/**
 * The game (stage 5.2, restyled in D.7): the master's narration in the story's font (with prose markup, ADR 67),
 * the characters' lines in bubbles under their names, the player's on the right; live streaming, cards.
 */
export default function GameScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const game = useGame(id);
  const headerHeight = useHeaderHeight();
  const list = useRef<FlatList<FeedRow>>(null);
  const theme = useTheme();
  const title = useSaveTitle(id);

  const adapting = game.phase === 'adapting';
  const canType = game.ready && !game.busy && (game.phase === 'playing' || (adapting && !!game.proposal?.canClarify));
  const submit = (value: string) => {
    if (adapting) game.clarify(value);
    else game.send(value);
  };

  // Every proposal keeps its cards, choosable until the hero is chosen (stage T.3).
  const items = useMemo(
    () => feedRows({ lines: game.lines, costs: game.costs, pending: game.pending, live: game.live }),
    [game.lines, game.pending, game.live, game.costs],
  );

  const spent = useAllowanceSpent();
  const limit = spent ? 'prototype' : limitOf(game.error);
  const footer = limit ? (
    <View style={styles.footer}>
      <LimitNotice limit={limit} />
    </View>
  ) : (
    <View style={styles.footer}>
      {game.busy && (game.thinking || game.live.length === 0) ? (
        <View style={styles.spinner}>
          <FoolBells label={game.phase === 'adapting' ? 'The story is fitting your hero…' : 'The master is thinking…'} />
        </View>
      ) : null}
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
      <Stack.Screen options={{ title, headerRight: () => <ReportButton send={game.report} /> }} />
      <SafeAreaView style={styles.column} edges={['bottom']}>
        {/* `padding` on Android too (stage T.1): with edge-to-edge (Android 15+) the system no
            longer shrinks the window for the keyboard, so the field would stay under it. */}
        <KeyboardAvoidingView style={styles.screen} behavior="padding" keyboardVerticalOffset={headerHeight}>
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
                  <Bubble speaker={item.speaker} name={item.name} text={item.text} streaming={item.streaming} />
                  {item.cards ? <CardCarousel cards={item.cards} disabled={game.busy || !adapting} onChoose={game.choose} /> : null}
                </View>
              )
            }
          />
          {(game.phase === 'playing' || adapting) && limit !== 'prototype' && limit !== 'budget' ? (
            <View style={[styles.inputRow, { borderTopColor: theme.line }]}>
              {adapting && game.proposal ? <ChangesLeft round={game.proposal.round} maxRounds={game.proposal.maxRounds} /> : null}
              <ChatInput
                editable={canType}
                placeholder={adapting ? (game.proposal?.canClarify ? 'Ask for changes…' : 'Choose a card') : 'What do you do?'}
                onSend={submit}
              />
            </View>
          ) : null}
        </KeyboardAvoidingView>
      </SafeAreaView>
    </ThemedView>
  );
}

function Bubble({ speaker, name, text, streaming }: { speaker: string; name?: string; text: string; streaming?: boolean }) {
  const theme = useTheme();
  const player = speaker === 'player';
  const npc = isActorChannel(speaker);
  if (!player && !npc) {
    return (
      <View style={styles.narration}>
        <Prose text={text} streaming={streaming} />
      </View>
    );
  }
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
      <Prose text={text} type={player ? 'body' : 'story'} streaming={streaming} />
    </View>
  );
}

/**
 * The adaptation's limit (stage T.2): three changes in all (four proposals), said plainly, and how
 * many are left. A proposal saved before T.2 has no `maxRounds`: the old three.
 */
function ChangesLeft({ round, maxRounds = 3 }: { round: number; maxRounds?: number }) {
  const changes = maxRounds - 1;
  const left = Math.max(0, maxRounds - round);
  return (
    <ThemedText type="caption" themeColor="inkMuted">
      {left > 0
        ? `You have ${changes} attempts in all to change your character · ${left} left`
        : 'No changes left: choose one of the characters above.'}
    </ThemedText>
  );
}

type Limit = 'prototype' | 'daily' | 'budget';

/**
 * A spent allowance (protocol `error` codes, ADR 66): the prototype's for good, or today's; or
 * the team's own budget for the models — the shared key cannot pay any more.
 */
function limitOf(error: string | undefined): Limit | undefined {
  if (!error) return undefined;
  if (error.includes('budget_exhausted')) return 'budget';
  if (error.includes('prototype_limit')) return 'prototype';
  if (error.includes('limit_exceeded')) return 'daily';
  return undefined;
}

/**
 * The guest's prototype allowance is already spent (ADR 66): say so when the game opens, before
 * the player types a move that would only fail.
 */
function useAllowanceSpent(): boolean {
  const [spent, setSpent] = useState(false);
  useEffect(() => {
    let active = true;
    void ensureSignedIn()
      .then(() => getCloud().auth.me())
      .then((me) => active && setSpent(me.totalLimitUsd !== null && me.spentUsd >= me.totalLimitUsd))
      .catch(() => {}); // offline: the cloud still refuses, and the notice comes then
    return () => {
      active = false;
    };
  }, []);
  return spent;
}

/** What the player sees instead of an error when the allowance is spent; asks for an email. */
const NOTICES: Record<Limit, { title: string; text: string }> = {
  prototype: {
    title: 'The prototype ends here',
    text: 'You have played every turn this prototype allows. Thank you for trying foolore!',
  },
  daily: { title: 'That is all for today', text: 'Today’s allowance is spent. The story will wait for you tomorrow.' },
  budget: {
    title: 'The ink has run dry',
    text:
      'This is an early prototype, and the budget our team set aside for the AI models has run out. ' +
      'Thank you for playing so much! Join the waitlist and we will write to you when foolore is released.',
  },
};

function LimitNotice({ limit }: { limit: Limit }) {
  const notice = NOTICES[limit];
  return (
    <Card>
      <ThemedText type="title">{notice.title}</ThemedText>
      <ThemedText type="story">{notice.text}</ThemedText>
      <WaitlistForm source="limit" intro="Leave an email and we will tell you when the full version is out." />
    </Card>
  );
}

/** The header names the game ("The Brass Vault — Wren"). */
function useSaveTitle(saveId: string): string {
  const [title, setTitle] = useState('');
  useEffect(() => {
    let active = true;
    void getLibrary()
      .then((library) => library.repository.getSave(saveId))
      .then((save) => active && setTitle(save?.title.split(' — ')[0] ?? ''))
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [saveId]);
  return title;
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  // A reading column on wide screens (the web version, a tablet), as on the other pages.
  column: { flex: 1, alignSelf: 'center', maxWidth: MaxContentWidth, width: '100%' },
  feed: { gap: Spacing.three, padding: Spacing.three },
  narration: { paddingHorizontal: Spacing.one },
  footer: { gap: Spacing.two },
  bubble: { borderRadius: Radius.large, borderWidth: StyleSheet.hairlineWidth * 2, gap: Spacing.one, maxWidth: '88%', padding: Spacing.three },
  player: { alignSelf: 'flex-end' },
  other: { alignSelf: 'flex-start' },
  spinner: { alignSelf: 'flex-start', padding: Spacing.two },
  inputRow: {
    borderTopWidth: StyleSheet.hairlineWidth,
    gap: Spacing.two,
    paddingHorizontal: Spacing.three,
    paddingVertical: Spacing.two,
  },
});
