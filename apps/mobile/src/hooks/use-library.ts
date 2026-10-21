import { useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';

import type { StoryRecord } from '@/core/library/app-db';
import type { CatalogStory } from '@/core/library/catalog';
import { ensureBundledStory } from '@/platform/bundled-story';
import { getCatalog, getLibrary } from '@/platform/services';

/** One cover of the library: a story on the device, in the cloud, or both. */
export interface LibraryStory {
  id: string;
  title: string;
  tagline: string;
  /** On the device (null: only in the cloud, downloaded when played). */
  local: StoryRecord | null;
  /** In the cloud's catalogue (null: offline, or a story imported by hand). */
  cloud: CatalogStory | null;
}

/**
 * The library (stages D.5, D.7): the stories on the device — the shipped one first imported
 * (ADR 64) — and those in the cloud's catalogue, one entry each, by title. Offline it is the
 * device's stories alone. Reloads whenever the screen comes into focus.
 */
export function useLibrary(): { stories: LibraryStory[]; error: string; loading: boolean } {
  const [local, setLocal] = useState<{ record: StoryRecord; tagline: string }[]>([]);
  const [cloud, setCloud] = useState<CatalogStory[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useFocusEffect(
    useCallback(() => {
      let active = true;
      void ensureBundledStory()
        .catch((e) => active && setError(e instanceof Error ? e.message : String(e)))
        .then(getLibrary)
        .then(async (library) => {
          const records = await library.repository.listStories();
          const metas = await Promise.all(records.map((r) => library.storyMeta(r.id).catch(() => undefined)));
          if (active) setLocal(records.map((record, i) => ({ record, tagline: metas[i]?.tagline ?? '' })));
        })
        .catch((e) => active && setError(e instanceof Error ? e.message : String(e)))
        .finally(() => active && setLoading(false));
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

  const stories: LibraryStory[] = [
    ...local.map(({ record, tagline }) => {
      const entry = cloud.find((c) => c.id === record.id) ?? null;
      return { id: record.id, title: record.title, tagline: tagline || entry?.tagline || '', local: record, cloud: entry };
    }),
    ...cloud
      .filter((entry) => !local.some((l) => l.record.id === entry.id))
      .map((entry) => ({ id: entry.id, title: entry.title, tagline: entry.tagline, local: null, cloud: entry })),
  ].sort((a, b) => a.title.localeCompare(b.title));

  return { stories, error, loading };
}

/** "In the cloud · 1.3 MB" for a story not on the device yet. */
export function cloudNote(story: LibraryStory): string | undefined {
  return story.local || !story.cloud ? undefined : `In the cloud · ${(story.cloud.size / 1_000_000).toFixed(1)} MB`;
}
