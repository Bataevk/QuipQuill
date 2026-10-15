import { File } from 'expo-file-system';
import { useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';

import { ThemedText } from '@/components/themed-text';
import { Button, Row, Screen } from '@/components/ui';
import type { StoryRecord } from '@/core/library/app-db';
import { StoryFormatError } from '@/core/story/story-meta';
import { pickedFilePath, removeFile } from '@/platform/expo-files';
import { getLibrary } from '@/platform/services';

/** Import a .qqstory and list the library (stage 3.2, moved here in 5.1). */
export default function StoriesScreen() {
  const [stories, setStories] = useState<StoryRecord[]>([]);
  const [status, setStatus] = useState('');

  const load = useCallback(async () => setStories(await (await getLibrary()).repository.listStories()), []);

  useEffect(() => {
    let active = true;
    getLibrary()
      .then((library) => library.repository.listStories())
      .then((list) => active && setStories(list))
      .catch((error) => active && setStatus(`Could not open the library: ${String(error)}`));
    return () => {
      active = false;
    };
  }, []);

  const importFrom = useCallback(
    async (path: string) => {
      try {
        setStatus('Importing…');
        const { story, replaced } = await (await getLibrary()).importer.import(path);
        setStatus(`${replaced ? 'Updated' : 'Imported'} “${story.title}” (${story.vectorIndex})`);
        await load();
      } catch (error) {
        setStatus(error instanceof StoryFormatError ? error.message : `Import failed: ${String(error)}`);
      }
    },
    [load],
  );

  const pickAndImport = async () => {
    const picked = await File.pickFileAsync();
    if (picked.canceled) return;
    const { path, temporary } = await pickedFilePath(picked.result);
    try {
      await importFrom(path);
    } finally {
      if (temporary) removeFile(path);
    }
  };

  // Dev only: quipquill://stories?importPath=<absolute path> imports without the system picker
  // (simulator automation; see docs/notes/dev-setup.md).
  const { importPath } = useLocalSearchParams<{ importPath?: string }>();
  useEffect(() => {
    if (__DEV__ && importPath) void Promise.resolve(importPath).then(importFrom);
  }, [importPath, importFrom]);

  return (
    <Screen>
      <Button title="Import story" onPress={pickAndImport} />
      {status ? <ThemedText themeColor="textSecondary">{status}</ThemedText> : null}
      {stories.map((story) => (
        <Row key={story.id} title={story.title} subtitle={`Imported ${story.importedAt.slice(0, 10)}`} />
      ))}
      {stories.length === 0 ? <ThemedText themeColor="textSecondary">No stories yet.</ThemedText> : null}
    </Screen>
  );
}
