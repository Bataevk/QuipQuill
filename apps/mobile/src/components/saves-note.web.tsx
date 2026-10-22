import { StyleSheet } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Card } from '@/components/ui';

/**
 * The web version keeps games only in this browser (ADR 69): clearing the site's data deletes
 * them, and Safari clears it after a week without a visit (ADR 39).
 */
export function SavesNote() {
  return (
    <Card>
      <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
        YOUR GAMES
      </ThemedText>
      <ThemedText type="caption">
        Games and heroes are saved in this browser only. Clearing its site data deletes them, and Safari may clear it
        after a week without a visit.
      </ThemedText>
    </Card>
  );
}

const styles = StyleSheet.create({
  eyebrow: { letterSpacing: 1.2 },
});
