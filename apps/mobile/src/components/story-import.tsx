import { File } from 'expo-file-system';
import { router, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { StyleSheet } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Button, Card } from '@/components/ui';
import { StoryFormatError } from '@/core/story/story-meta';
import { pickedFilePath, removeFile } from '@/platform/expo-files';
import { getLibrary } from '@/platform/services';

/**
 * Import a .qqstory from a file (stage 3.2). In dev builds quipquill://profile?importPath=<absolute
 * path> imports without the system picker (simulator automation; see docs/notes/dev-setup.md).
 */
export function StoryImport() {
  const [status, setStatus] = useState('');

  const importFrom = useCallback(async (path: string) => {
    try {
      setStatus('Importing…');
      const { story, replaced } = await (await getLibrary()).importer.import(path);
      setStatus(`${replaced ? 'Updated' : 'Imported'} “${story.title}” (${story.vectorIndex})`);
    } catch (error) {
      setStatus(error instanceof StoryFormatError ? error.message : `Import failed: ${String(error)}`);
    }
  }, []);

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

  const { importPath } = useLocalSearchParams<{ importPath?: string }>();
  useEffect(() => {
    if (!__DEV__ || !importPath) return;
    // The tab keeps its params: clear it, or every visit to the profile imports again.
    router.setParams({ importPath: undefined });
    void Promise.resolve(importPath).then(importFrom);
  }, [importPath, importFrom]);

  return (
    <Card>
      <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
        STORY FILES
      </ThemedText>
      <Button title="Import a story file" onPress={() => void pickAndImport()} />
      {status ? (
        <ThemedText type="caption" themeColor="inkMuted">
          {status}
        </ThemedText>
      ) : null}
    </Card>
  );
}

const styles = StyleSheet.create({
  eyebrow: { letterSpacing: 1.2 },
});
