import { StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Button, Screen } from '@/components/ui';
import { Spacing } from '@/constants/theme';

/**
 * The web version in a second tab (stage W.2, ADR 69): the browser's storage belongs to the
 * first tab, so this one covers the app (whose screens cannot load their data) and shows only
 * how to get back to the game. A layer over the navigator, like the boot screen: the router
 * needs its navigator mounted to keep the URL.
 */
export function StorageBusy() {
  return (
    <View style={StyleSheet.absoluteFill}>
      <Screen title="Already open" scroll={false}>
        <View style={styles.body}>
          <ThemedText>foolore is open in another tab of this browser. Your games live there.</ThemedText>
          <ThemedText themeColor="inkMuted">Close that tab, then reload this one.</ThemedText>
          <Button title="Reload" variant="primary" onPress={() => globalThis.location?.reload()} />
        </View>
      </Screen>
    </View>
  );
}

const styles = StyleSheet.create({
  body: { gap: Spacing.three },
});
