import { describe, expect, it } from 'vitest';

import type { TranscriptLine } from '@/core/agent/transcript';

import { feedRows, type ProposalCards } from './feed-rows';

const hero = (name: string) => ({ name, race: 'human', age: 30, description: `${name}.` });
const THREE = [hero('A'), hero('B'), hero('C')];
let seq = 0;
const line = (speaker: TranscriptLine['speaker'], text: string, extra: Record<string, unknown> = {}): TranscriptLine => ({
  seq: ++seq,
  speaker,
  text,
  extra,
  ts: '2026-10-22T00:00:00.000Z',
});
const proposal = (round: number) => line('gm', `Round ${round}.`, { kind: 'character_proposal', round, options: THREE, canClarify: true, maxRounds: 4 });
const cardsOf = (lines: TranscriptLine[]) =>
  feedRows({ lines, costs: [], live: [] }).flatMap((r) => (r.kind === 'bubble' && r.cards ? [r.cards] : []));

describe('feedRows: the character cards (stage T.3)', () => {
  it('keeps every proposal with its cards, all choosable until the hero is chosen', () => {
    const cards = cardsOf([proposal(1), line('player', 'Older.'), proposal(2), line('player', 'Taller.'), proposal(3)]);
    expect(cards.map((c) => [c.round, c.options.length, c.open, c.latest])).toEqual([
      [1, 3, true, false],
      [2, 3, true, false],
      [3, 3, true, true],
    ]);
    expect(cards.every((c) => c.chosen === undefined)).toBe(true);
  });

  it('closes every proposal once a hero is chosen and marks the card picked', () => {
    const cards = cardsOf([
      proposal(1),
      line('player', 'Older.'),
      proposal(2),
      line('player', 'B, human', { kind: 'character_choice', choice: 'alternative 1', round: 1, index: 1 }),
      line('gm', 'The story begins.'),
    ]);
    expect(cards.map((c: ProposalCards) => [c.round, c.open, c.chosen])).toEqual([
      [1, false, 1],
      [2, false, undefined],
    ]);
  });

  it('reads a pick saved before stage T.3: the card of the proposal just before it', () => {
    const cards = cardsOf([
      proposal(1),
      line('player', 'Older.'),
      proposal(2),
      line('player', 'C, human', { kind: 'character_choice', choice: 'alternative 2' }),
    ]);
    expect(cards.map((c) => c.chosen)).toEqual([undefined, 2]);
  });

  it('puts costs after their line, then the pending move, the live bubbles and the footer', () => {
    const first = line('gm', 'Hello.');
    const rows = feedRows({
      lines: [first],
      costs: [{ afterSeq: first.seq, text: '$0.001' }],
      pending: 'I wave.',
      live: [{ requestId: 'r1', channel: 'npc:mags', speaker: 'Mags', text: 'Hm.' }],
    });
    expect(rows.map((r) => r.kind)).toEqual(['bubble', 'cost', 'bubble', 'bubble', 'footer']);
    expect(rows[3]).toMatchObject({ speaker: 'npc:mags', name: 'Mags', streaming: true });
  });
});
