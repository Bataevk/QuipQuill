import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';

import { Cover } from '@/components/cover';
import { FoolProgress } from '@/components/fool-loader';
import { OnboardingPage } from '@/components/onboarding';
import { ThemedText } from '@/components/themed-text';
import { Button, ErrorText, errorText } from '@/components/ui';
import { Radius, Spacing } from '@/constants/theme';
import type { Character } from '@/core/library/app-db';
import { cloudNote, useLibrary, type LibraryStory } from '@/hooks/use-library';
import { useTheme } from '@/hooks/use-theme';
import { finishOnboarding } from '@/platform/onboarding';
import { getCatalog, getLibrary } from '@/platform/services';

/**
 * The last step (D.6): pick a story and play your new hero in it. A story only in the cloud is
 * downloaded first (FoolProgress); then the game opens and the master fits the hero to the world.
 */
export default function StoryScreen() {
  const { hero: heroId } = useLocalSearchParams<{ hero: string }>();
  const library = useLibrary();
  const theme = useTheme();
  const [hero, setHero] = useState<Character>();
  const [chosen, setChosen] = useState<string>();
  const [download, setDownload] = useState<number>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    void getLibrary()
      .then((l) => l.repository.getCharacter(heroId))
      .then(setHero)
      .catch(() => {});
  }, [heroId]);

  // The only story: chosen already.
  const selected = library.stories.find((s) => s.id === (chosen ?? (library.stories.length === 1 ? library.stories[0].id : undefined)));

  const play = async (story: LibraryStory) => {
    if (!hero) return;
    setBusy(true);
    setError('');
    try {
      if (!story.local && story.cloud) {
        setDownload(0);
        await (await getCatalog()).install(story.cloud, { onProgress: setDownload });
      }
      const save = await (await getLibrary()).saves.create({ storyId: story.id, characterId: hero.id, title: `${story.title} — ${hero.name}` });
      await finishOnboarding();
      // The main pages under the game, so "Back" from the game lands on Stories.
      router.dismissAll();
      router.replace('/');
      router.push(`/game/${save.id}`);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setDownload(undefined);
      setBusy(false);
    }
  };

  const footer =
    download !== undefined ? (
      <FoolProgress progress={download} label="Downloading the story…" size={96} />
    ) : (
      <Button
        title={selected && hero ? `Play as ${hero.name}` : 'Choose a story'}
        variant="primary"
        busy={busy}
        disabled={!selected || !hero}
        onPress={() => selected && void play(selected)}
      />
    );

  return (
    <OnboardingPage step={3} footer={footer}>
      <ThemedText type="title">Choose a story</ThemedText>
      {library.stories.map((story) => (
        <Pressable
          key={story.id}
          accessibilityRole="button"
          accessibilityState={{ selected: selected?.id === story.id }}
          onPress={() => setChosen(story.id)}
          style={[styles.choice, { borderColor: selected?.id === story.id ? theme.accent : 'transparent' }]}>
          <Cover title={story.title} subtitle={[story.tagline, cloudNote(story)].filter(Boolean).join('\n') || undefined} />
        </Pressable>
      ))}
      {!library.loading && library.stories.length === 0 ? (
        <ThemedText themeColor="inkMuted">No stories could be found. Check the connection and come back to this step.</ThemedText>
      ) : null}
      <View>
        <ErrorText>{error || library.error}</ErrorText>
      </View>
    </OnboardingPage>
  );
}

const styles = StyleSheet.create({
  choice: { borderRadius: Radius.medium + 3, borderWidth: 3, padding: Spacing.half },
});
