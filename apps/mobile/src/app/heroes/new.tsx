import { router, useLocalSearchParams } from 'expo-router';
import { StyleSheet, View } from 'react-native';

import { useHeroWizard } from '@/components/hero-wizard';
import { Button, Screen } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { newId, nowIso } from '@/core/util/ids';
import { getLibrary } from '@/platform/services';

/**
 * A new hero (stage D.7), step by step like the first launch (D.6). Opened from a story's
 * "Play as" (`?story=<id>`), the game starts right after; from the Heroes tab, it goes back to the list.
 * The buttons follow the question (not pinned to the bottom), so the keyboard never covers them.
 */
export default function NewHeroScreen() {
  const { story } = useLocalSearchParams<{ story?: string }>();
  const wizard = useHeroWizard({
    createTitle: story ? 'Create and play' : 'Create hero',
    onCreate: async (sheet) => {
      const library = await getLibrary();
      const id = newId();
      await library.repository.addCharacter({ id, ...sheet, createdAt: nowIso() });
      if (!story) return router.back();
      const record = await library.repository.getStory(story);
      if (!record) return router.back();
      const game = await library.saves.create({ storyId: story, characterId: id, title: `${record.title} — ${sheet.name}` });
      router.dismissTo(`/game/${game.id}`);
    },
  });

  return (
    <Screen>
      {wizard.body}
      <View style={styles.actions}>
        {wizard.footer}
        {!wizard.atStart ? <Button title="Back" variant="ghost" onPress={wizard.back} /> : null}
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  actions: { gap: Spacing.two, marginTop: Spacing.two },
});
