import { File } from 'expo-file-system';
import { useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { Pressable, ScrollView, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { MaxContentWidth, Spacing } from '@/constants/theme';
import type { StoryRecord } from '@/core/library/app-db';
import { StoryFormatError } from '@/core/story/story-meta';
import { toPath } from '@/platform/expo-files';
import { getLibrary } from '@/platform/services';

// Stage 3.2: import a .qqstory and list the library. Real screens arrive in stage 5.1.
export default function HomeScreen() {
  const [stories, setStories] = useState<StoryRecord[]>([]);
  const [status, setStatus] = useState('');

  const loadStories = useCallback(
    async () => (await getLibrary()).repository.listStories(),
    [],
  );

  useEffect(() => {
    let active = true;
    loadStories()
      .then((list) => active && setStories(list))
      .catch((error) => active && setStatus(`Could not open the library: ${String(error)}`));
    return () => {
      active = false;
    };
  }, [loadStories]);

  const importFrom = useCallback(
    async (path: string) => {
      try {
        setStatus('Importing…');
        const library = await getLibrary();
        const { story, replaced } = await library.importer.import(path);
        setStatus(`${replaced ? 'Updated' : 'Imported'} “${story.title}” (${story.vectorIndex})`);
        setStories(await loadStories());
      } catch (error) {
        setStatus(
          error instanceof StoryFormatError ? error.message : `Import failed: ${String(error)}`,
        );
      }
    },
    [loadStories],
  );

  const pickAndImport = async () => {
    const picked = await File.pickFileAsync();
    if (!picked.canceled) await importFrom(toPath(picked.result.uri));
  };

  // Dev only: quipquill://?importPath=<absolute path> imports without the system picker
  // (simulator automation; see docs/notes/dev-setup.md).
  const { importPath } = useLocalSearchParams<{ importPath?: string }>();
  useEffect(() => {
    if (__DEV__ && importPath) void Promise.resolve(importPath).then(importFrom);
  }, [importPath, importFrom]);

  return (
    <ThemedView style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ThemedText type="title">QuipQuill</ThemedText>
        <Pressable accessibilityRole="button" onPress={pickAndImport} style={styles.button}>
          <ThemedText type="linkPrimary">Import story</ThemedText>
        </Pressable>
        {status ? <ThemedText themeColor="textSecondary">{status}</ThemedText> : null}
        <ScrollView contentContainerStyle={styles.list}>
          {stories.map((story) => (
            <ThemedText key={story.id} type="subtitle">
              {story.title}
            </ThemedText>
          ))}
          {stories.length === 0 ? (
            <ThemedText themeColor="textSecondary">No stories yet.</ThemedText>
          ) : null}
        </ScrollView>
      </SafeAreaView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'center',
  },
  safeArea: {
    flex: 1,
    maxWidth: MaxContentWidth,
    gap: Spacing.three,
    paddingHorizontal: Spacing.four,
    paddingTop: Spacing.five,
  },
  button: {
    paddingVertical: Spacing.two,
  },
  list: {
    gap: Spacing.two,
  },
});
