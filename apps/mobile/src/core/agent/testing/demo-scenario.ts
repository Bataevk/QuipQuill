/**
 * The stage 4.1 scenario on the demo story (The Brass Vault): look around → go → take →
 * check the inventory. Shared by the core tests (Node) and the on-device check (Hermes).
 */
import type { BaseMessage } from '@langchain/core/messages';

import type { ChatStore } from '../chat-store';
import type { GmAgent } from '../gm-agent';
import { runTurn, type TurnResult } from '../run-turn';
import type { World } from '../../world/world';
import type { ScriptedChatModel, Step } from './scripted-model';

/** Text of the last tool message the model was given, or of the system prompt. */
export const lastToolText = (messages: BaseMessage[]) =>
  messages.findLast((m) => m.getType() === 'tool')?.text ?? '';
export const systemText = (messages: BaseMessage[]) =>
  messages[0]?.getType() === 'system' ? messages[0].text : '';

export const DEMO_TURNS: { input: string; steps: Step[] }[] = [
  {
    input: 'I look around the tavern.',
    steps: [
      { toolCalls: [{ name: 'inspect', args: { target: 'loc:rusty-gear-tavern' } }] },
      (messages) => ({
        text: lastToolText(messages).includes('Mags Holloway')
          ? 'Smoke curls under the low beams. Mags Holloway watches you over her mug.'
          : 'The tavern is strangely empty.',
      }),
    ],
  },
  {
    input: 'I climb the stairs to Lamplight Row.',
    steps: [
      { toolCalls: [{ name: 'move', args: { to: 'Lamplight Row' } }] },
      { text: 'Gas lamps hiss along the wet cobbles of Lamplight Row.' },
    ],
  },
  {
    input: 'I pick up the newspaper from the gutter.',
    steps: [
      { toolCalls: [{ name: 'take', args: { item: 'evening newspaper' } }] },
      { text: 'You fold the damp evening paper into your coat.' },
    ],
  },
  {
    input: 'What am I carrying?',
    steps: [
      // No tool needed: the inventory is in the world state of the system prompt.
      (messages) => ({
        text: /Inventory: .*discarded evening newspaper/.test(systemText(messages))
          ? 'Only the discarded evening newspaper, folded in your coat.'
          : 'Your pockets are empty.',
      }),
    ],
  },
];

export async function playDemoScenario(input: {
  model: ScriptedChatModel;
  agent: GmAgent;
  world: World;
  chats: ChatStore;
}): Promise<TurnResult[]> {
  const results: TurnResult[] = [];
  for (const turn of DEMO_TURNS) {
    input.model.push(...turn.steps);
    results.push(await runTurn({ ...input, input: turn.input }));
  }
  return results;
}
