import { router, useLocalSearchParams } from 'expo-router';

import { HeroForm, type HeroSheet } from '@/components/hero-form';
import { Screen } from '@/components/ui';
import { newId, nowIso } from '@/core/util/ids';
import { getLibrary } from '@/platform/services';

/**
 * A new hero (stage D.7). Opened from a story's "Play as" (`?story=<id>`), the game starts right
 * after saving; from the Heroes tab, it goes back to the list.
 */
export default function NewHeroScreen() {
  const { story } = useLocalSearchParams<{ story?: string }>();

  const save = async (sheet: HeroSheet) => {
    const library = await getLibrary();
    const id = newId();
    await library.repository.addCharacter({ id, ...sheet, createdAt: nowIso() });
    if (!story) return router.back();
    const record = await library.repository.getStory(story);
    if (!record) return router.back();
    const game = await library.saves.create({ storyId: story, characterId: id, title: `${record.title} — ${sheet.name}` });
    router.dismissTo(`/game/${game.id}`);
  };

  return (
    <Screen>
      <HeroForm submitTitle={story ? 'Create and play' : 'Create hero'} onSubmit={save} />
    </Screen>
  );
}
