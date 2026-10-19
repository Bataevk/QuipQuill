// Stage 4.1 check: the game master agent on Hermes with op-sqlite (scripted model, no cloud).
// To rerun on a device: copy to src/app/agent-check.tsx, start Metro, import the demo story
// and open quipquill://agent-check (docs/notes/dev-setup.md).
import { useEffect, useState } from 'react';
import { ScrollView, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Spacing } from '@/constants/theme';
import { ChatStore } from '@/core/agent/chat-store';
import { TranscriptStore } from '@/core/agent/transcript';
import { createGmAgent } from '@/core/agent/gm-agent';
import { playDemoScenario } from '@/core/agent/testing/demo-scenario';
import { ScriptedChatModel } from '@/core/agent/testing/scripted-model';
import { readStoryMeta } from '@/core/story/story-meta';
import { openVectorIndex } from '@/core/vector/open-index';
import { World } from '@/core/world/world';
import { getLibrary } from '@/platform/services';

const DEMO_STORY_ID = 'd09514c5-6b96-48a4-ba57-2cc4b4b68f55';

async function runAgentCheck() {
  const library = await getLibrary();
  if (!(await library.repository.getStory(DEMO_STORY_ID))) throw new Error('Import the demo story first');
  const save = await library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 'agent check' });
  const db = await library.saves.open(save.id);
  try {
    const vectors = await openVectorIndex(db);
    const world = await World.open(db, { vectors });
    const meta = await readStoryMeta(db);
    const started = await world.startSession({
      storyId: DEMO_STORY_ID,
      character: { id: null, name: 'Tester', description: 'A tester.' },
      startLocation: meta.startLocation,
    });
    if (!started.ok) throw new Error(started.error);
    const model = new ScriptedChatModel([]);
    const agent = createGmAgent({ model, world, meta });
    const t0 = Date.now();
    const results = await playDemoScenario({
      model,
      agent,
      world,
      chats: new ChatStore(db),
      transcript: new TranscriptStore(db),
    });
    const ms = Date.now() - t0;
    const scene = await world.scene();
    const stored = await new ChatStore(db).load('gm');
    const summary = {
      vectorIndex: vectors.constructor.name,
      events: results.map((r) => r.events.map((e) => e.kind)),
      replies: results.map((r) => r.reply),
      location: scene.location?.id,
      inventory: scene.inventory.map((e) => e.id),
      storedMessages: stored.length,
      modelCalls: model.calls.length,
      ms,
    };
    const ok =
      JSON.stringify(summary.events) === JSON.stringify([[], ['player.moved'], ['item.taken'], []]) &&
      summary.location === 'loc:lamplight-row' &&
      summary.inventory.join() === 'doc:discarded-evening-newspaper' &&
      summary.replies[3].startsWith('Only the discarded evening newspaper') &&
      summary.storedMessages === 14;
    return { ok, ...summary };
  } finally {
    await db.close();
    await library.saves.delete(save.id);
  }
}

export default function AgentCheckScreen() {
  const [report, setReport] = useState('Running…');

  useEffect(() => {
    runAgentCheck()
      .catch((error: Error) => ({ ok: false, error: `${error.name}: ${error.message}\n${error.stack ?? ''}` }))
      .then((result) => {
        console.log(`AGENT_CHECK ${JSON.stringify(result)}`);
        setReport(JSON.stringify(result, null, 2));
      });
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
