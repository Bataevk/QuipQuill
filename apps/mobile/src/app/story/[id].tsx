import { router, Stack, useFocusEffect, useLocalSearchParams } from 'expo-router';
import { useCallback, useRef, useState } from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';

import { Cover } from '@/components/cover';
import { FoolProgress } from '@/components/fool-loader';
import { sheetLine } from '@/components/format';
import { SaveCard } from '@/components/save-card';
import { ThemedText } from '@/components/themed-text';
import { Button, ErrorText, Row, Screen, Sheet, errorText } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import type { Character, SaveRecord, StoryRecord } from '@/core/library/app-db';
import { StoryCatalog, type CatalogStory, type CatalogState } from '@/core/library/catalog';
import type { StoryHero, StoryMeta } from '@/core/story/story-meta';
import { nowIso } from '@/core/util/ids';
import { getCatalog, getLibrary } from '@/platform/services';

/** The world's summary is long; the page shows its start so "New game" stays in sight. */
const ABOUT_LINES = 5;

const megabytes = (bytes: number) => `${(bytes / 1_000_000).toFixed(1)} MB`;

/**
 * A story's page (stage D.7, the cloud in D.5): the cover, what the world is, your games in it,
 * and "New game", which asks which hero to play — one written for the story (ADR 62) or your own.
 * A story only in the cloud is downloaded first; a newer build replaces the copy when the page
 * opens (saves keep their own copies, ADR 24).
 */
export default function StoryScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [local, setLocal] = useState<StoryRecord>();
  const [meta, setMeta] = useState<StoryMeta>();
  const [entry, setEntry] = useState<CatalogStory>();
  const [saves, setSaves] = useState<SaveRecord[]>([]);
  /** Every character the saves may point to (the story's heroes too); `heroes` — the player's own. */
  const [characters, setCharacters] = useState<Character[]>([]);
  const heroes = characters.filter((c) => c.storyId === null);
  const [loaded, setLoaded] = useState(false);
  const [picking, setPicking] = useState(false);
  const [starting, setStarting] = useState<string>();
  const [download, setDownload] = useState<{ share: number; label: string }>();
  const [error, setError] = useState('');
  const [aboutOpen, setAboutOpen] = useState(false);
  const updating = useRef(false);

  const state: CatalogState | undefined = entry ? StoryCatalog.state(entry, local) : local ? 'installed' : undefined;

  const load = useCallback(async () => {
    const library = await getLibrary();
    const [record, all, characters] = await Promise.all([
      library.repository.getStory(id),
      library.repository.listSaves(),
      library.repository.listCharacters(),
    ]);
    setLocal(record);
    setMeta(record ? await library.storyMeta(id) : undefined);
    setSaves(all.filter((s) => s.storyId === id));
    setCharacters(characters);
    setLoaded(true);
    // The cloud may be out of reach: the page still works with the copy on the device.
    const online = await getCatalog()
      .then((catalog) => catalog.list())
      .catch(() => undefined);
    const found = online?.find((s) => s.id === id);
    setEntry(found);
    return { record, found };
  }, [id]);

  const install = useCallback(
    async (story: CatalogStory, label: string) => {
      setError('');
      setDownload({ share: 0, label });
      try {
        await (await getCatalog()).install(story, { onProgress: (share) => setDownload({ share, label }) });
        await load();
        return true;
      } catch (e) {
        setError(errorText(e));
        return false;
      } finally {
        setDownload(undefined);
      }
    },
    [load],
  );

  useFocusEffect(
    useCallback(() => {
      load()
        .then(({ record, found }) => {
          // A newer build in the cloud: bring the copy up to date once, while the page is open.
          if (found && record && StoryCatalog.state(found, record) === 'update' && !updating.current) {
            updating.current = true;
            void install(found, 'Updating the story…');
          }
        })
        .catch((e) => setError(errorText(e)));
    }, [load, install]),
  );

  const title = meta?.title ?? entry?.title ?? '';
  const tagline = meta?.tagline || entry?.tagline || undefined;
  const summary = meta?.worldSummary || entry?.summary || '';
  const storyHeroes = meta?.heroes ?? [];

  const begin = async (hero: Character) => {
    const save = await (await getLibrary()).saves.create({ storyId: id, characterId: hero.id, title: `${title} — ${hero.name}` });
    setPicking(false);
    router.push(`/game/${save.id}`);
  };

  const play = async (key: string, hero: () => Promise<Character>) => {
    setStarting(key);
    setError('');
    try {
      await begin(await hero());
    } catch (e) {
      setError(errorText(e));
    } finally {
      setStarting(undefined);
    }
  };

  const playStoryHero = (hero: StoryHero) =>
    play(`story:${hero.key}`, async () => (await getLibrary()).repository.upsertStoryHero(id, hero, nowIso()));

  const newGame = async () => {
    if (state === 'available' && entry) {
      if (await install(entry, 'Downloading the story…')) setPicking(true);
      return;
    }
    setPicking(true);
  };

  const createHero = () => {
    setPicking(false);
    router.push({ pathname: '/heroes/new', params: { story: id } });
  };

  const heroName = (save: SaveRecord) => characters.find((h) => h.id === save.characterId)?.name;

  if (loaded && !local && !entry && !download) {
    return (
      <Screen>
        <Stack.Screen options={{ title: '' }} />
        <ThemedText themeColor="inkMuted">This story is not on the device, and the cloud cannot be reached right now.</ThemedText>
        <ErrorText>{error}</ErrorText>
      </Screen>
    );
  }

  return (
    <Screen>
      <Stack.Screen options={{ title }} />
      <Cover title={title} subtitle={tagline} large />
      {download ? (
        <FoolProgress progress={download.share} label={download.label} size={120} />
      ) : (
        <Button
          title={state === 'available' && entry ? `Download and play · ${megabytes(entry.size)}` : 'New game'}
          variant="primary"
          disabled={!title || (!local && !entry)}
          onPress={() => void newGame()}
        />
      )}
      {summary ? (
        <View style={styles.about}>
          <ThemedText type="story" numberOfLines={aboutOpen ? undefined : ABOUT_LINES}>
            {summary}
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
        <ScrollView contentContainerStyle={styles.list}>
          {storyHeroes.length > 0 ? (
            <>
              <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
                FROM THIS STORY
              </ThemedText>
              {storyHeroes.map((hero) => (
                <Row
                  key={hero.key}
                  title={starting === `story:${hero.key}` ? `${hero.name} — opening…` : hero.name}
                  subtitle={hero.blurb || sheetLine(hero) || undefined}
                  onPress={starting ? undefined : () => void playStoryHero(hero)}
                />
              ))}
            </>
          ) : null}
          {heroes.length > 0 ? (
            <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
              YOUR HEROES
            </ThemedText>
          ) : storyHeroes.length === 0 ? (
            <ThemedText themeColor="inkMuted">You have no heroes yet. Make one — the story will fit them into its world.</ThemedText>
          ) : null}
          {heroes.map((hero) => (
            <Row
              key={hero.id}
              title={starting === hero.id ? `${hero.name} — opening…` : hero.name}
              subtitle={sheetLine(hero) || undefined}
              onPress={starting ? undefined : () => void play(hero.id, async () => hero)}
            />
          ))}
          <Button title="Create a new hero" variant={heroes.length || storyHeroes.length ? 'ghost' : 'primary'} onPress={createHero} />
        </ScrollView>
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
