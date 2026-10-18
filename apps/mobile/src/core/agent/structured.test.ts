import { describe, expect, it } from 'vitest';
import { z } from 'zod';

import { CHARACTER_RESPONSE_FORMAT, Proposal } from './character';
import { Notes, NOTE_ENTITY_RESPONSE_FORMAT } from './entity-memory';
import { Changes, GENERATOR_RESPONSE_FORMAT } from './generator';
import { GUARD_RESPONSE_FORMAT, Verdict } from './middleware/guard';
import { parseJsonAnswer, withTimeout } from './structured';

const FORMATS = [
  { name: 'character', format: CHARACTER_RESPONSE_FORMAT, schema: Proposal },
  { name: 'generator', format: GENERATOR_RESPONSE_FORMAT, schema: Changes },
  { name: 'guard', format: GUARD_RESPONSE_FORMAT, schema: Verdict },
  { name: 'noteEntity', format: NOTE_ENTITY_RESPONSE_FORMAT, schema: Notes },
];

/**
 * The JSON schema zod derives from a parsing schema, reduced to what `response_format` carries:
 * no defaults and no length or range limits (the answer is checked by zod anyway), a nullable
 * type as `type: [t, 'null']`.
 */
function wireShape(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(wireShape);
  if (!node || typeof node !== 'object') return node;
  const { $schema, default: _d, minLength, exclusiveMinimum, minimum, maximum, anyOf, ...rest } = node as Record<
    string,
    unknown
  >;
  void [$schema, _d, minLength, exclusiveMinimum, minimum, maximum];
  if (Array.isArray(anyOf) && anyOf.length === 2 && anyOf.some((s) => (s as { type?: string }).type === 'null')) {
    const other = anyOf.find((s) => (s as { type?: string }).type !== 'null') as { type: string };
    return { ...rest, type: [other.type, 'null'] };
  }
  return Object.fromEntries(Object.entries(rest).map(([key, value]) => [key, wireShape(value)]));
}

describe('response formats', () => {
  // What goes to the provider must not change by accident (stage R.1): the snapshot was taken
  // before the formats moved to the shared helpers. As JSON text, so the key order (what the
  // cloud forwards) is checked too.
  it('stay as they were', () => {
    expect(JSON.stringify(Object.fromEntries(FORMATS.map((f) => [f.name, f.format])))).toMatchSnapshot();
  });

  // The hand-written schema and the zod schema that reads the answer describe the same shape.
  it.each(FORMATS)('$name matches the schema that reads its answer', ({ format, schema }) => {
    expect(format.json_schema.schema).toEqual(wireShape(z.toJSONSchema(schema, { io: 'output' })));
  });
});

describe('parseJsonAnswer', () => {
  const Shape = z.object({ n: z.number(), tag: z.string().default('none') });

  it('reads plain JSON and applies defaults', () => {
    expect(parseJsonAnswer(Shape, ' {"n": 1} ')).toEqual({ n: 1, tag: 'none' });
  });

  it('tolerates a ```json fence', () => {
    expect(parseJsonAnswer(Shape, '```json\n{"n": 2, "tag": "x"}\n```')).toEqual({ n: 2, tag: 'x' });
  });

  it('throws on text that is not JSON or not of the shape', () => {
    expect(() => parseJsonAnswer(Shape, 'no')).toThrow();
    expect(() => parseJsonAnswer(Shape, '{"n": "one"}')).toThrow();
  });
});

describe('withTimeout', () => {
  it('aborts when the caller aborts', () => {
    const caller = new AbortController();
    const signal = withTimeout(60_000, caller.signal);
    caller.abort();
    expect(signal.aborted).toBe(true);
  });

  it('aborts at the limit', async () => {
    const signal = withTimeout(5);
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(signal.aborted).toBe(true);
  });
});
