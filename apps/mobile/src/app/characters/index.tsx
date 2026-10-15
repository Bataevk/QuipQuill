import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';

import { ThemedText } from '@/components/themed-text';
import { characterLine } from '@/components/format';
import { Button, Row, Screen } from '@/components/ui';
import type { Character } from '@/core/library/app-db';
import { getLibrary } from '@/platform/services';

/** Characters live outside playthroughs (ADR 12). */
export default function CharactersScreen() {
  const [characters, setCharacters] = useState<Character[]>([]);

  useFocusEffect(
    useCallback(() => {
      let active = true;
      void getLibrary()
        .then((library) => library.repository.listCharacters())
        .then((list) => active && setCharacters(list));
      return () => {
        active = false;
      };
    }, []),
  );

  return (
    <Screen>
      <Button title="New character" onPress={() => router.push('/characters/new')} />
      {characters.map((c) => (
        <Row key={c.id} title={c.name} subtitle={characterLine(c)} />
      ))}
      {characters.length === 0 ? <ThemedText themeColor="textSecondary">No characters yet.</ThemedText> : null}
    </Screen>
  );
}
