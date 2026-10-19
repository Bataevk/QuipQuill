import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Alert, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Button, ErrorText, Row, Screen, errorText } from '@/components/ui';
import type { SaveRecord, StoryRecord } from '@/core/library/app-db';
import { Spacing } from '@/constants/theme';
import { getCloud, getLibrary } from '@/platform/services';

/** "15 Oct, 19:31" in the device's time zone (records keep UTC ISO-8601). */
function localTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
}

/** Asks before an action that cannot be undone. */
function confirm(title: string, message: string, action: string, run: () => void) {
  Alert.alert(title, message, [
    { text: 'Cancel', style: 'cancel' },
    { text: action, style: 'destructive', onPress: run },
  ]);
}

/** Home: where to go, and the playthroughs: continue, start over, delete (stage 5.3). */
export default function HomeScreen() {
  const [saves, setSaves] = useState<SaveRecord[]>([]);
  const [stories, setStories] = useState<Map<string, StoryRecord>>(new Map());
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    const library = await getLibrary();
    const [list, storyList] = await Promise.all([library.repository.listSaves(), library.repository.listStories()]);
    return { list, stories: new Map(storyList.map((s) => [s.id, s])) };
  }, []);

  useFocusEffect(
    useCallback(() => {
      let active = true;
      load()
        .then((next) => {
          if (!active) return;
          setSaves(next.list);
          setStories(next.stories);
        })
        .catch((e) => active && setError(errorText(e)));
      return () => {
        active = false;
      };
    }, [load]),
  );

  /** Runs a save action, then shows the list again. */
  const act = async (work: () => Promise<unknown>) => {
    setError('');
    try {
      await work();
      const next = await load();
      setSaves(next.list);
      setStories(next.stories);
    } catch (e) {
      setError(errorText(e));
    }
  };

  const startOver = (save: SaveRecord) =>
    confirm('Start over?', `“${save.title}” begins again from the start. Everything played so far is lost.`, 'Start over', () =>
      void act(async () => {
        await (await getLibrary()).saves.restart(save.id);
        router.push(`/game/${save.id}`);
      }),
    );

  const remove = (save: SaveRecord) =>
    confirm('Delete this game?', `“${save.title}” will be deleted for good.`, 'Delete', () =>
      void act(async () => (await getLibrary()).saves.delete(save.id)),
    );

  const signOut = async () => {
    const { auth, socket } = getCloud();
    socket.stop();
    await auth.logout();
    router.replace('/login');
  };

  return (
    <Screen>
      <Button title="New game" variant="primary" onPress={() => router.push('/new-game')} />
      <Button title="Stories" onPress={() => router.push('/stories')} />
      <Button title="Characters" onPress={() => router.push('/characters')} />
      <ThemedText type="label">Continue</ThemedText>
      {saves.map((save) => (
        <View key={save.id} style={styles.save}>
          <Row
            title={save.title}
            subtitle={`${stories.get(save.storyId)?.title ?? 'Unknown story'} · ${localTime(save.updatedAt)}`}
            onPress={() => router.push(`/game/${save.id}`)}
          />
          <View style={styles.actions}>
            <View style={styles.action}>
              <Button title="Start over" onPress={() => startOver(save)} />
            </View>
            <View style={styles.action}>
              <Button title="Delete" onPress={() => remove(save)} />
            </View>
          </View>
        </View>
      ))}
      {saves.length === 0 ? <ThemedText themeColor="inkMuted">No games yet.</ThemedText> : null}
      <ErrorText>{error}</ErrorText>
      <Button title="Sign out" variant="ghost" onPress={signOut} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  save: { gap: Spacing.one },
  actions: { flexDirection: 'row', gap: Spacing.one },
  action: { flex: 1 },
});
