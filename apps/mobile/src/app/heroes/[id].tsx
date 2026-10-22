import { router, Stack, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';

import { confirmAction } from '@/components/confirm';
import { HeroForm, type HeroSheet } from '@/components/hero-form';
import { ThemedText } from '@/components/themed-text';
import { Button, ErrorText, Screen, errorText } from '@/components/ui';
import type { Character } from '@/core/library/app-db';
import { getLibrary } from '@/platform/services';

/** Edit or delete a hero (stage D.7). Games already started keep the hero they adapted. */
export default function HeroScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [hero, setHero] = useState<Character | null>();
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    void getLibrary()
      .then((library) => library.repository.getCharacter(id))
      .then((found) => active && setHero(found ?? null))
      .catch((e) => active && setError(errorText(e)));
    return () => {
      active = false;
    };
  }, [id]);

  const save = async (sheet: HeroSheet) => {
    await (await getLibrary()).repository.updateCharacter({ id, ...sheet });
    router.back();
  };

  const remove = () =>
    confirmAction('Delete this hero?', `${hero?.name ?? 'This hero'} will be deleted for good.`, 'Delete', () =>
      void getLibrary()
        .then((library) => library.repository.deleteCharacter(id))
        .then(() => router.back())
        .catch((e) => setError(errorText(e))),
    );

  return (
    <Screen>
      <Stack.Screen options={{ title: hero?.name ?? '' }} />
      {hero ? <HeroForm initial={hero} submitTitle="Save" onSubmit={save} /> : null}
      {hero === null ? <ThemedText themeColor="inkMuted">This hero no longer exists.</ThemedText> : null}
      {hero ? <Button title="Delete hero" variant="ghost" onPress={remove} /> : null}
      <ErrorText>{error}</ErrorText>
    </Screen>
  );
}
