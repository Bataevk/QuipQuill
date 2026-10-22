import { router } from 'expo-router';
import { StyleSheet, View } from 'react-native';

import { confirmAction as confirm } from '@/components/confirm';
import { localTime } from '@/components/format';
import { ThemedText } from '@/components/themed-text';
import { Button, Card } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import type { SaveRecord } from '@/core/library/app-db';
import { getLibrary } from '@/platform/services';

/**
 * One playthrough with its actions (stage 5.3, moved to the story page in D.7): continue,
 * start over, delete. `onChanged` reloads the list; `onError` shows what went wrong.
 */
export function SaveCard({
  save,
  hero,
  onChanged,
  onError,
}: {
  save: SaveRecord;
  hero?: string;
  onChanged: () => void;
  onError: (error: unknown) => void;
}) {
  const act = (work: () => Promise<unknown>) =>
    void work()
      .then(onChanged)
      .catch(onError);

  const startOver = () =>
    confirm('Start over?', `“${save.title}” begins again from the start. Everything played so far is lost.`, 'Start over', () =>
      act(async () => {
        await (await getLibrary()).saves.restart(save.id);
        router.push(`/game/${save.id}`);
      }),
    );

  const remove = () =>
    confirm('Delete this game?', `“${save.title}” will be deleted for good.`, 'Delete', () =>
      act(async () => (await getLibrary()).saves.delete(save.id)),
    );

  return (
    <Card>
      <View style={styles.head}>
        <ThemedText type="label">{hero ?? save.title}</ThemedText>
        <ThemedText type="caption" themeColor="inkMuted">
          {localTime(save.updatedAt)}
        </ThemedText>
      </View>
      <Button title="Continue" variant="primary" onPress={() => router.push(`/game/${save.id}`)} />
      <View style={styles.actions}>
        <View style={styles.action}>
          <Button title="Start over" onPress={startOver} />
        </View>
        <View style={styles.action}>
          <Button title="Delete" variant="ghost" onPress={remove} />
        </View>
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  head: { gap: Spacing.half },
  actions: { flexDirection: 'row', gap: Spacing.two },
  action: { flex: 1 },
});
