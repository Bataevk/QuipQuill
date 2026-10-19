import { router, Stack, useFocusEffect, useLocalSearchParams } from 'expo-router';
import { useCallback, useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { Cover } from '@/components/cover';
import { sheetLine } from '@/components/format';
import { SaveCard } from '@/components/save-card';
import { ThemedText } from '@/components/themed-text';
import { Button, ErrorText, Row, Screen, Sheet, errorText } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import type { Character, SaveRecord } from '@/core/library/app-db';
import type { StoryMeta } from '@/core/story/story-meta';
import { getLibrary } from '@/platform/services';

/** The world's summary is long; the page shows its start so "New game" stays in sight. */
const ABOUT_LINES = 5;

/**
 * A story's page (stage D.7): the cover, what the world is, your games in it, and "New game",
 * which asks which hero to play. Heroes written by the story's author join the list in D.5.
 */
export default function StoryScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [meta, setMeta] = useState<StoryMeta>();
  const [saves, setSaves] = useState<SaveRecord[]>([]);
  const [heroes, setHeroes] = useState<Character[]>([]);
  const [picking, setPicking] = useState(false);
  const [starting, setStarting] = useState<string>();
  const [error, setError] = useState('');
  const [aboutOpen, setAboutOpen] = useState(false);

  const load = useCallback(async () => {
    const library = await getLibrary();
    const [m, all, h] = await Promise.all([library.storyMeta(id), library.repository.listSaves(), library.repository.listCharacters()]);
    setMeta(m);
    setSaves(all.filter((s) => s.storyId === id));
    setHeroes(h);
  }, [id]);

  useFocusEffect(
    useCallback(() => {
      load().catch((e) => setError(errorText(e)));
    }, [load]),
  );

  const play = async (hero: Character) => {
    if (!meta) return;
    setStarting(hero.id);
    setError('');
    try {
      const save = await (await getLibrary()).saves.create({ storyId: id, characterId: hero.id, title: `${meta.title} — ${hero.name}` });
      setPicking(false);
      router.push(`/game/${save.id}`);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setStarting(undefined);
    }
  };

  const createHero = () => {
    setPicking(false);
    router.push({ pathname: '/heroes/new', params: { story: id } });
  };

  const heroName = (save: SaveRecord) => heroes.find((h) => h.id === save.characterId)?.name;

  return (
    <Screen>
      <Stack.Screen options={{ title: meta?.title ?? '' }} />
      <Cover title={meta?.title ?? ''} large />
      <Button title="New game" variant="primary" disabled={!meta} onPress={() => setPicking(true)} />
      {meta?.worldSummary ? (
        <View style={styles.about}>
          <ThemedText type="story" numberOfLines={aboutOpen ? undefined : ABOUT_LINES}>
            {meta.worldSummary}
          </ThemedText>
          {!aboutOpen ? <Button title="Read more" variant="ghost" onPress={() => setAboutOpen(true)} /> : null}
        </View>
      ) : null}
      <ErrorText>{error}</ErrorText>

      {saves.length > 0 ? (
        <View style={styles.section}>
          <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
            YOUR GAMES
          </ThemedText>
          {saves.map((save) => (
            <SaveCard key={save.id} save={save} hero={heroName(save)} onChanged={() => void load()} onError={(e) => setError(errorText(e))} />
          ))}
        </View>
      ) : null}

      <Sheet visible={picking} onClose={() => setPicking(false)} title="Play as">
        {heroes.length > 0 ? (
          <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
            YOUR HEROES
          </ThemedText>
        ) : (
          <ThemedText themeColor="inkMuted">You have no heroes yet. Make one — the story will fit them into its world.</ThemedText>
        )}
        <View style={styles.list}>
          {heroes.map((hero) => (
            <Row
              key={hero.id}
              title={starting === hero.id ? `${hero.name} — opening…` : hero.name}
              subtitle={sheetLine(hero) || undefined}
              onPress={starting ? undefined : () => void play(hero)}
            />
          ))}
        </View>
        <Button title="Create a new hero" variant={heroes.length ? 'ghost' : 'primary'} onPress={createHero} />
      </Sheet>
    </Screen>
  );
}

const styles = StyleSheet.create({
  section: { gap: Spacing.two },
  about: { gap: Spacing.one },
  list: { gap: Spacing.two },
  eyebrow: { letterSpacing: 1.2 },
});
