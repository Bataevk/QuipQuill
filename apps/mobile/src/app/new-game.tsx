import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';

import { characterLine } from '@/components/format';
import { ThemedText } from '@/components/themed-text';
import { Button, ErrorText, Row, Screen, errorText } from '@/components/ui';
import type { Character, StoryRecord } from '@/core/library/app-db';
import { getLibrary } from '@/platform/services';

/** A story + a character → a new save, then the game screen. */
export default function NewGameScreen() {
  const [stories, setStories] = useState<StoryRecord[]>([]);
  const [characters, setCharacters] = useState<Character[]>([]);
  const [story, setStory] = useState<string>();
  const [character, setCharacter] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useFocusEffect(
    useCallback(() => {
      let active = true;
      void getLibrary().then(async (library) => {
        const [s, c] = await Promise.all([library.repository.listStories(), library.repository.listCharacters()]);
        if (!active) return;
        setStories(s);
        setCharacters(c);
      });
      return () => {
        active = false;
      };
    }, []),
  );

  const start = async () => {
    const chosenStory = stories.find((s) => s.id === story);
    const chosenCharacter = characters.find((c) => c.id === character);
    if (!chosenStory || !chosenCharacter) return;
    setBusy(true);
    try {
      const save = await (await getLibrary()).saves.create({
        storyId: chosenStory.id,
        characterId: chosenCharacter.id,
        title: `${chosenStory.title} — ${chosenCharacter.name}`,
      });
      router.replace(`/game/${save.id}`);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Screen>
      <ThemedText type="label">Story</ThemedText>
      {stories.map((s) => (
        <Row key={s.id} title={s.title} selected={s.id === story} onPress={() => setStory(s.id)} />
      ))}
      {stories.length === 0 ? <Button title="Import a story first" onPress={() => router.push('/stories')} /> : null}
      <ThemedText type="label">Character</ThemedText>
      {characters.map((c) => (
        <Row key={c.id} title={c.name} subtitle={characterLine(c)} selected={c.id === character} onPress={() => setCharacter(c.id)} />
      ))}
      {characters.length === 0 ? <Button title="Create a character first" onPress={() => router.push('/characters/new')} /> : null}
      <Button title="Start" variant="primary" onPress={start} busy={busy} disabled={!story || !character} />
      <ErrorText>{error}</ErrorText>
    </Screen>
  );
}
