import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, expect, it } from 'vitest';

import {
  actorFailureOf,
  actorLine,
  actorOf,
  blockedResult,
  currentTurn,
  guardReply,
  inputMessage,
  isBlocked,
  isGuardReply,
  masterReplies,
} from './messages';

const MAGS = { id: 'npc:mags', name: 'Mags' };

describe('current turn', () => {
  it('starts at the last player or system input; summaries do not start one', () => {
    const history = [
      inputMessage('look', 'player', 1),
      new AIMessage('A bar.'),
      inputMessage('talk to Mags', 'player', 2),
      new AIMessage('She looks up.'),
    ];
    expect(currentTurn(history)).toEqual(history.slice(2));
    expect(currentTurn([inputMessage('Earlier…', 'summary'), new AIMessage('x')])).toHaveLength(2);
  });
});

describe('markers', () => {
  it("an entity's line keeps who said it, and why it failed", () => {
    expect(actorOf(actorLine(MAGS, '"No."'))).toEqual(MAGS);
    expect(actorFailureOf(actorLine(MAGS, '"No."'))).toBeUndefined();
    expect(actorFailureOf(actorLine(MAGS, '', 'timeout'))).toBe('timeout');
    expect(actorOf(new AIMessage('narration'))).toBeUndefined();
  });

  it('keeps the stored form of earlier stages', () => {
    expect(actorLine(MAGS, 'x').additional_kwargs).toEqual({ qq_actor: MAGS });
    expect(guardReply('Nope.', 'claims a sword').response_metadata).toMatchObject({ guard: 'reject' });
    expect(blockedResult({ id: 'c1', name: 'inspect' }, 'Error').additional_kwargs).toEqual({ qq_blocked: true });
  });

  it('tells the Guard reply and blocked results from others', () => {
    expect(isGuardReply(guardReply('Nope.', ''))).toBe(true);
    expect(isGuardReply(new AIMessage('Fine.'))).toBe(false);
    expect(isBlocked(blockedResult({ id: 'c1', name: 'inspect' }, 'Error'))).toBe(true);
    expect(isBlocked(new ToolMessage({ tool_call_id: 'c1', content: 'ok' }))).toBe(false);
    expect(isBlocked(undefined)).toBe(false);
  });

  it("the master's replies leave out entities' lines and empty calls", () => {
    const turn = [
      inputMessage('hi', 'player', 1),
      new AIMessage({ content: '', tool_calls: [{ id: 'c1', name: 'react_entity', args: {} }] }),
      actorLine(MAGS, '"Hello."'),
      new AIMessage('She nods.'),
    ];
    expect(masterReplies(turn).map((m) => m.text)).toEqual(['She nods.']);
  });
});
