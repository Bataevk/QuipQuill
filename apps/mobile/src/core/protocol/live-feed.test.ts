import { describe, expect, it } from 'vitest';

import { LiveFeed } from './live-feed';
import type { ServerEvent } from './messages';

const routed = (requestId: string, channel = 'gm') => ({ v: 1 as const, requestId, sessionId: 's', channel });
const delta = (requestId: string, seq: number, text: string, channel?: string): ServerEvent => ({
  ...routed(requestId, channel),
  type: 'delta',
  seq,
  text,
});
const done = (requestId: string, content: string, channel?: string): ServerEvent => ({
  ...routed(requestId, channel),
  type: 'done',
  message: { role: 'assistant', content },
  finishReason: 'stop',
  usage: { input: 1, output: 1, cacheRead: 0, costUsd: 0 },
});

describe('LiveFeed', () => {
  it('builds bubbles of story channels in the order generations started', () => {
    const feed = new LiveFeed();
    feed.begin();
    expect(feed.apply(delta('g1', 0, 'You reach'))).toBe('new');
    expect(feed.apply({ ...routed('n1', 'npc:mags'), type: 'delta', seq: 0, text: '"Hm."', meta: { speaker: 'Mags' } })).toBe('new');
    expect(feed.apply(delta('g1', 1, ' for it.'))).toBe('changed');
    expect(feed.apply(delta('g1', 1, ' for it.'))).toBeUndefined(); // duplicate seq
    expect(feed.apply(delta('x', 0, '{"verdict"', 'guard'))).toBeUndefined();
    expect(feed.bubbles()).toEqual([
      { requestId: 'g1', channel: 'gm', speaker: undefined, text: 'You reach for it.' },
      { requestId: 'n1', channel: 'npc:mags', speaker: 'Mags', text: '"Hm."' },
    ]);
  });

  it('ignores done and error repeated after resume for generations it never streamed', () => {
    const feed = new LiveFeed();
    feed.begin();
    expect(feed.apply(done('old-1', 'Text from ten minutes ago.'))).toBeUndefined();
    expect(feed.apply({ v: 1, type: 'error', requestId: 'old-2', sessionId: 's', channel: 'gm', code: 'timeout', statusKey: 'error.timeout', retryable: true })).toBeUndefined();
    expect(feed.apply({ ...routed('g2'), type: 'status', status: 'thinking' })).toBeUndefined();
    // a generation in flight comes back through replay, which replaces its text
    expect(feed.apply({ ...routed('g2'), type: 'replay', text: 'Everything so far' })).toBe('new');
    expect(feed.apply(done('g2', 'Everything so far, and the end.'))).toBe('changed');
    expect(feed.bubbles().map((b) => b.text)).toEqual(['Everything so far, and the end.']);
  });

  it('accepts nothing outside an action', () => {
    const feed = new LiveFeed();
    expect(feed.apply(delta('g1', 0, 'early'))).toBeUndefined();
    feed.begin();
    feed.apply(delta('g1', 0, 'during'));
    feed.end();
    expect(feed.bubbles()).toEqual([]);
    expect(feed.apply(delta('g1', 1, 'late'))).toBeUndefined();
  });

  it('keeps a speaker found later', () => {
    const feed = new LiveFeed();
    feed.begin();
    feed.apply(delta('n1', 0, '"Hey."', 'npc:pip'));
    expect(feed.speakerKnown('n1')).toBe(false);
    feed.setSpeaker('n1', 'Pip Quillon');
    expect(feed.bubbles()[0]).toMatchObject({ speaker: 'Pip Quillon', channel: 'npc:pip' });
    expect(feed.channelOf('n1')).toBe('npc:pip');
  });
});
