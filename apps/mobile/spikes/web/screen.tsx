// Stage W.1 spike screen (web). To rerun: copy to src/app/spike.tsx, fix the relative imports
// from './' and '../' to '../../spikes/web/' and '../../spikes/', copy opfs-worker.mjs to
// public/sqlite/, run `npm run web` and open http://localhost:8081/spike.
import { Asset } from 'expo-asset';
import { useEffect, useState } from 'react';
import { ScrollView, StyleSheet, Text } from 'react-native';

import { runAgentSpike } from '../agent';

// eslint-disable-next-line @typescript-eslint/no-require-imports
const DEMO_STORY = require('../../../../stories/brass-vault/brass-vault.qqstory');

type WorkerReply = { ok: boolean; report: unknown; error?: string };

async function runOpfsSpike(): Promise<WorkerReply & { storyFetchMs: number }> {
  const t0 = performance.now();
  const story = await (await fetch(Asset.fromModule(DEMO_STORY).uri)).arrayBuffer();
  const storyFetchMs = Math.round(performance.now() - t0);
  const worker = new Worker('/sqlite/opfs-worker.mjs', { type: 'module' });
  const reply = await new Promise<WorkerReply>((resolve) => {
    worker.onmessage = ({ data }) => resolve(data as WorkerReply);
    worker.onerror = (event) => resolve({ ok: false, report: null, error: event.message });
    worker.postMessage({ story }, [story]);
  });
  worker.terminate();
  return { ...reply, storyFetchMs };
}

export default function SpikeScreen() {
  const [report, setReport] = useState('Running…');

  useEffect(() => {
    void (async () => {
      const env = {
        userAgent: navigator.userAgent,
        opfs: typeof navigator.storage?.getDirectory === 'function',
        locks: typeof navigator.locks?.request === 'function',
        persisted: await navigator.storage?.persisted?.(),
        estimate: await navigator.storage?.estimate?.(),
      };
      const opfs = await runOpfsSpike();
      const agent = await runAgentSpike().catch((error: unknown) => ({ error: String(error) }));
      const result = { env, opfs, agent };
      console.log(`SPIKE_RESULT ${JSON.stringify(result)}`);
      // Browsers the agent cannot drive post here (a local collector, see spikes/web/README.md).
      void fetch('http://localhost:8092/', { method: 'POST', body: JSON.stringify(result) }).catch(() => {});
      setReport(JSON.stringify(result, null, 2));
    })();
  }, []);

  return (
    <ScrollView contentContainerStyle={styles.body}>
      <Text selectable style={styles.text}>
        {report}
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  body: { padding: 16 },
  text: { fontFamily: 'monospace', fontSize: 12 },
});
