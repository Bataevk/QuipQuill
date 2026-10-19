import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { describe, expect, it } from 'vitest';

import { promptMessage, promptPart } from '@/core/agent/cloud-prompt';
import { fromDone, toWireMessages } from '@/core/net/wire-format';
import type { DoneEvent } from '@/core/protocol/messages';
import { StreamText } from '@/core/protocol/stream-text';

const routed = { v: 1 as const, requestId: 'r1', sessionId: 's1', channel: 'gm' };

describe('wire format', () => {
  it('converts a tool round trip to OpenAI messages, keeping cache breakpoints', () => {
    const wire = toWireMessages([
      new SystemMessage({
        content: [{ type: 'text', text: 'Rules.', cache_control: { type: 'ephemeral' } }],
      }),
      new HumanMessage('I walk north.'),
      new AIMessage({ content: '', tool_calls: [{ id: 'c1', name: 'move', args: { to: 'hall' } }] }),
      new ToolMessage({ content: 'Moved to hall.', tool_call_id: 'c1' }),
    ]);
    expect(wire).toEqual([
      { role: 'system', content: [{ type: 'text', text: 'Rules.', cache_control: { type: 'ephemeral' } }] },
      { role: 'user', content: 'I walk north.' },
      {
        role: 'assistant',
        content: null,
        tool_calls: [{ id: 'c1', type: 'function', function: { name: 'move', arguments: '{"to":"hall"}' } }],
      },
      { role: 'tool', content: 'Moved to hall.', tool_call_id: 'c1' },
    ]);
  });

  it('sends a prompt template part as it is: the cloud fills it in (ADR 57)', () => {
    const part = promptPart('gm', { STORY_STYLE: 'Noir.' }, { type: 'ephemeral', ttl: '1h' });
    const wire = toWireMessages([promptMessage(part, { type: 'text', text: 'Scene.' })]);
    expect(wire).toEqual([{ role: 'system', content: [part, { type: 'text', text: 'Scene.' }] }]);
  });

  it('keeps a breakpoint on an assistant text or a tool result as parts (ADR 46)', () => {
    const cached = (text: string) => [{ type: 'text', text, cache_control: { type: 'ephemeral' as const } }];
    const wire = toWireMessages([
      new AIMessage({ content: cached('"Below the bank."') }),
      new ToolMessage({ content: cached('Noted.'), tool_call_id: 'c1' }),
      new ToolMessage({ content: 'plain', tool_call_id: 'c2' }),
    ]);
    expect(wire).toEqual([
      { role: 'assistant', content: cached('"Below the bank."') },
      { role: 'tool', content: cached('Noted.'), tool_call_id: 'c1' },
      { role: 'tool', content: 'plain', tool_call_id: 'c2' },
    ]);
  });

  it('turns done into an AIMessage with parsed tool calls, usage and bad arguments flagged', () => {
    const done: DoneEvent = {
      ...routed,
      type: 'done',
      message: {
        role: 'assistant',
        content: 'Heading north.',
        tool_calls: [
          { id: 'c1', type: 'function', function: { name: 'move', arguments: '{"to":"hall"}' } },
          { id: 'c2', type: 'function', function: { name: 'take', arguments: '{"item":' } },
        ],
      },
      finishReason: 'tool_calls',
      usage: { input: 1200, output: 80, cacheRead: 1000, cacheWrite: 0, costUsd: 0.0001 },
    };
    const message = fromDone(done);
    expect(message.content).toBe('Heading north.');
    expect(message.tool_calls).toEqual([{ id: 'c1', name: 'move', args: { to: 'hall' }, type: 'tool_call' }]);
    expect(message.invalid_tool_calls?.[0]).toMatchObject({ id: 'c2', name: 'take' });
    expect(message.usage_metadata).toMatchObject({ input_tokens: 1200, output_tokens: 80, total_tokens: 1280 });
    expect(message.response_metadata).toMatchObject({ finish_reason: 'tool_calls', cost_usd: 0.0001 });
  });
});

describe('StreamText', () => {
  it('drops duplicate deltas, lets replay replace the text, and ignores other requests', () => {
    const text = new StreamText('r1');
    text.apply({ ...routed, type: 'delta', seq: 0, text: 'Hel' });
    text.apply({ ...routed, type: 'delta', seq: 0, text: 'Hel' }); // duplicate
    text.apply({ ...routed, requestId: 'other', type: 'delta', seq: 1, text: 'XX' });
    expect(text.text).toBe('Hel');

    text.apply({ ...routed, type: 'replay', text: 'Hello, tr' }); // after a reconnect
    text.apply({ ...routed, type: 'delta', seq: 5, text: 'aveller' });
    expect(text.text).toBe('Hello, traveller');

    text.apply({
      ...routed,
      type: 'done',
      message: { role: 'assistant', content: 'Hello, traveller.' },
      finishReason: 'stop',
      usage: { input: 1, output: 1, cacheRead: 0, costUsd: 0 },
    });
    expect([text.text, text.done]).toEqual(['Hello, traveller.', true]);
  });
});
