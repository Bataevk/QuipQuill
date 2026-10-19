// Stage 3.1 spike screen. To rerun on a device: copy to src/app/spike.tsx, fix the
// relative imports below from './' to '../../spikes/', start Metro and open quipquill://spike.
import { useEffect, useState } from 'react';
import { ScrollView, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Spacing } from '@/constants/theme';

import { nativeGlobals } from './shims/hermes-globals';
import { runAgentSpike } from './agent';
import { runSqliteSpike } from './sqlite-vec';

export default function SpikeScreen() {
  const [report, setReport] = useState('Running…');

  useEffect(() => {
    (async () => {
      const sqlite = await runSqliteSpike();
      const agent = await runAgentSpike();
      const text = JSON.stringify({ nativeGlobals, sqlite, agent }, null, 2);
      console.log(`SPIKE_RESULT ${JSON.stringify({ nativeGlobals, sqlite, agent })}`);
      setReport(text);
    })();
  }, []);

  return (
    <ThemedView style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView>
          <ThemedText type="caption">{report}</ThemedText>
        </ScrollView>
      </SafeAreaView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  safeArea: { flex: 1, padding: Spacing.three },
});
