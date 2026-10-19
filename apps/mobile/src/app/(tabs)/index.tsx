import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { Cover } from '@/components/cover';
import { localTime } from '@/components/format';
import { ThemedText } from '@/components/themed-text';
import { ErrorText, Row, Screen, errorText } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import type { Character, SaveRecord, StoryRecord } from '@/core/library/app-db';
import { ensureBundledStory } from '@/platform/bundled-story';
import { getLibrary } from '@/platform/services';

/** How many recent games the top of the page offers to continue. */
const RECENT = 3;

/**
 * Stories (stage D.7): where you stopped on top, the library below. A story opens its page,
 * where you pick a hero. The demo story ships inside the app until the cloud catalogue (D.5).
 */
export default function StoriesScreen() {
  const [stories, setStories] = useState<StoryRecord[]>([]);
  const [saves, setSaves] = useState<SaveRecord[]>([]);
  const [heroes, setHeroes] = useState<Map<string, Character>>(new Map());
  const [error, setError] = useState('');

  useFocusEffect(
    useCallback(() => {
      let active = true;
      void ensureBundledStory()
        .catch((e) => active && setError(errorText(e)))
        .then(getLibrary)
        .then(async ({ repository }) => {
          const [s, sv, h] = await Promise.all([repository.listStories(), repository.listSaves(), repository.listCharacters()]);
          if (!active) return;
          setStories(s);
          setSaves(sv);
          setHeroes(new Map(h.map((c) => [c.id, c])));
        })
        .catch((e) => active && setError(errorText(e)));
      return () => {
        active = false;
      };
    }, []),
  );

  const titleOf = (id: string) => stories.find((s) => s.id === id)?.title ?? 'Unknown story';

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
        {stories.map((story) => (
          <Cover key={story.id} title={story.title} onPress={() => router.push(`/story/${story.id}`)} />
        ))}
        {stories.length === 0 ? (
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
