import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { Cover } from '@/components/cover';
import { localTime } from '@/components/format';
import { ThemedText } from '@/components/themed-text';
import { ErrorText, Row, Screen, errorText } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import type { Character, SaveRecord, StoryRecord } from '@/core/library/app-db';
import type { CatalogStory } from '@/core/library/catalog';
import { ensureBundledStory } from '@/platform/bundled-story';
import { getCatalog, getLibrary } from '@/platform/services';

/** How many recent games the top of the page offers to continue. */
const RECENT = 3;

/**
 * Stories (stage D.7): where you stopped on top, the library below — the stories on the device
 * and those in the cloud's catalogue (D.5), one cover each. A story opens its page, where you
 * pick a hero (a story only in the cloud is downloaded there). The demo story also ships inside
 * the app (ADR 64), so the library is never empty offline.
 */
export default function StoriesScreen() {
  const [stories, setStories] = useState<StoryRecord[]>([]);
  const [taglines, setTaglines] = useState<Map<string, string>>(new Map());
  const [cloud, setCloud] = useState<CatalogStory[]>([]);
  const [saves, setSaves] = useState<SaveRecord[]>([]);
  const [heroes, setHeroes] = useState<Map<string, Character>>(new Map());
  const [error, setError] = useState('');

  useFocusEffect(
    useCallback(() => {
      let active = true;
      void ensureBundledStory()
        .catch((e) => active && setError(errorText(e)))
        .then(getLibrary)
        .then(async (library) => {
          const { repository } = library;
          const [s, sv, h] = await Promise.all([repository.listStories(), repository.listSaves(), repository.listCharacters()]);
          if (!active) return;
          setStories(s);
          setSaves(sv);
          setHeroes(new Map(h.map((c) => [c.id, c])));
          const metas = await Promise.all(s.map((story) => library.storyMeta(story.id).catch(() => undefined)));
          if (active) setTaglines(new Map(metas.flatMap((m) => (m ? [[m.storyId, m.tagline] as const] : []))));
        })
        .catch((e) => active && setError(errorText(e)));
      // The catalogue after the device's own stories: offline, the library still shows them.
      void getCatalog()
        .then((catalog) => catalog.list())
        .then((list) => active && setCloud(list))
        .catch(() => {});
      return () => {
        active = false;
      };
    }, []),
  );

  const titleOf = (id: string) => stories.find((s) => s.id === id)?.title ?? 'Unknown story';
  const library = [
    ...stories.map((story) => ({ id: story.id, title: story.title, subtitle: taglines.get(story.id) || cloud.find((c) => c.id === story.id)?.tagline })),
    ...cloud
      .filter((entry) => !stories.some((s) => s.id === entry.id))
      .map((entry) => ({
        id: entry.id,
        title: entry.title,
        subtitle: [entry.tagline, `In the cloud · ${(entry.size / 1_000_000).toFixed(1)} MB`].filter(Boolean).join('\n'),
      })),
  ].sort((a, b) => a.title.localeCompare(b.title));

  return (
    <Screen>
      {saves.length > 0 ? (
        <View style={styles.section}>
          <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
            CONTINUE
          </ThemedText>
          {saves.slice(0, RECENT).map((save) => (
            <Row
              key={save.id}
              title={titleOf(save.storyId)}
              subtitle={`${(save.characterId && heroes.get(save.characterId)?.name) || save.title} · ${localTime(save.updatedAt)}`}
              onPress={() => router.push(`/game/${save.id}`)}
            />
          ))}
        </View>
      ) : null}

      <View style={styles.section}>
        <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
          LIBRARY
        </ThemedText>
        {library.map((story) => (
          <Cover key={story.id} title={story.title} subtitle={story.subtitle || undefined} onPress={() => router.push(`/story/${story.id}`)} />
        ))}
        {library.length === 0 ? (
          <ThemedText themeColor="inkMuted">
            No stories yet.
          </ThemedText>
        ) : null}
      </View>
      <ErrorText>{error}</ErrorText>
    </Screen>
  );
}

const styles = StyleSheet.create({
  section: { gap: Spacing.two },
  eyebrow: { letterSpacing: 1.2 },
});
