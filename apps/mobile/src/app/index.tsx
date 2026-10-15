import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';

import { ThemedText } from '@/components/themed-text';
import { Button, ErrorText, Row, Screen, errorText } from '@/components/ui';
import type { SaveRecord, StoryRecord } from '@/core/library/app-db';
import { getCloud, getLibrary } from '@/platform/services';

/** Home: where to go, and the playthroughs to continue (full save management — stage 5.3). */
export default function HomeScreen() {
  const [saves, setSaves] = useState<SaveRecord[]>([]);
  const [stories, setStories] = useState<Map<string, StoryRecord>>(new Map());
  const [error, setError] = useState('');

  useFocusEffect(
    useCallback(() => {
      let active = true;
      (async () => {
        const library = await getLibrary();
        const [list, storyList] = await Promise.all([library.repository.listSaves(), library.repository.listStories()]);
        if (!active) return;
        setSaves(list);
        setStories(new Map(storyList.map((s) => [s.id, s])));
      })().catch((e) => active && setError(errorText(e)));
      return () => {
        active = false;
      };
    }, []),
  );

  const signOut = async () => {
    const { auth, socket } = getCloud();
    socket.stop();
    await auth.logout();
    router.replace('/login');
  };

  return (
    <Screen>
      <Button title="New game" onPress={() => router.push('/new-game')} />
      <Button title="Stories" onPress={() => router.push('/stories')} />
      <Button title="Characters" onPress={() => router.push('/characters')} />
      <ThemedText type="smallBold">Continue</ThemedText>
      {saves.map((save) => (
        <Row
          key={save.id}
          title={save.title}
          subtitle={`${stories.get(save.storyId)?.title ?? 'Unknown story'} · ${save.updatedAt.slice(0, 16).replace('T', ' ')}`}
          onPress={() => router.push(`/game/${save.id}`)}
        />
      ))}
      {saves.length === 0 ? <ThemedText themeColor="textSecondary">No games yet.</ThemedText> : null}
      <ErrorText>{error}</ErrorText>
      <Button title="Sign out" onPress={signOut} />
    </Screen>
  );
}
