import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';

import { characterLine } from '@/components/format';
import { ThemedText } from '@/components/themed-text';
import { Button, ErrorText, Row, Screen, errorText } from '@/components/ui';
import type { Character } from '@/core/library/app-db';
import { getLibrary } from '@/platform/services';

/** Your heroes (stage D.7): shared by every story (ADR 12, 62); tap to edit or delete. */
export default function HeroesScreen() {
  const [heroes, setHeroes] = useState<Character[]>([]);
  const [error, setError] = useState('');

  useFocusEffect(
    useCallback(() => {
      let active = true;
      void getLibrary()
        .then((library) => library.repository.listCharacters())
        // The player's own; the heroes written for a story live on its page (ADR 62).
        .then((list) => active && setHeroes(list.filter((c) => c.storyId === null)))
        .catch((e) => active && setError(errorText(e)));
      return () => {
        active = false;
      };
    }, []),
  );

  return (
    <Screen>
      <Button title="New hero" variant="primary" onPress={() => router.push('/heroes/new')} />
      {heroes.map((hero) => (
        <Row key={hero.id} title={hero.name} subtitle={characterLine(hero) || undefined} onPress={() => router.push(`/heroes/${hero.id}`)} />
      ))}
      {heroes.length === 0 ? (
        <ThemedText themeColor="inkMuted">No heroes yet. A hero goes with you into any story, and each story fits them into its world.</ThemedText>
      ) : null}
      <ErrorText>{error}</ErrorText>
    </Screen>
  );
}
