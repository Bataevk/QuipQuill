import { describe, expect, it } from 'vitest';

import { parseLuck } from '@/core/story/story-meta';

import { luckChance, MISHAP_BELOW, rollLuck } from './luck';

const SEARCH = { attempt: 'search under the tables', likelihood: 'possible' as const, stakes: 'useful' as const };
const fixed = (...values: number[]) => () => values.shift() ?? 0.5;

describe('luck (ADR 47)', () => {
  it('a fair world keeps the base odds; luck moves them the more the higher the stakes', () => {
    expect(luckChance('possible', 'useful', 0.5)).toBeCloseTo(0.45);
    expect(luckChance('certain', 'trivial', 0.5)).toBeCloseTo(0.95);
    expect(luckChance('possible', 'trivial', 1)).toBeCloseTo(0.585);
    expect(luckChance('possible', 'valuable', 1)).toBeCloseTo(0.9);
    expect(luckChance('possible', 'valuable', 0)).toBe(0.02); // never quite impossible
    expect(luckChance('likely', 'trivial', 0)).toBeCloseTo(0.525); // a trifle is found almost anyway
    expect(luckChance('certain', 'valuable', 1)).toBe(0.98);
  });

  it('rolls in code: below the chance it works out, above it fails', () => {
    expect(rollLuck(SEARCH, 0.5, fixed(0.2)).outcome).toBe('success');
    expect(rollLuck(SEARCH, 0.5, fixed(0.9)).outcome).toBe('failure');
    expect(rollLuck(SEARCH, 0.5, fixed(0.2))).toMatchObject({ ...SEARCH, luck: 0.5, chance: 0.45, roll: 0.2 });
  });

  it('a mishap only in a very unlucky world, and the unluckier the more often', () => {
    // a fair world: a failure is only a failure, whatever the second draw
    expect(rollLuck(SEARCH, 0.5, fixed(0.99, 0)).outcome).toBe('failure');
    expect(rollLuck(SEARCH, MISHAP_BELOW, fixed(0.99, 0)).outcome).toBe('failure');
    // luck 0.1: a failure turns into a mishap 60% of the time
    expect(rollLuck(SEARCH, 0.1, fixed(0.99, 0.5)).outcome).toBe('mishap');
    expect(rollLuck(SEARCH, 0.1, fixed(0.99, 0.7)).outcome).toBe('failure');
    // luck 0: every failure is a mishap; a success stays a success
    expect(rollLuck(SEARCH, 0, fixed(0.99, 0.99)).outcome).toBe('mishap');
    expect(rollLuck({ ...SEARCH, likelihood: 'certain', stakes: 'trivial' }, 0, fixed(0.1)).outcome).toBe('success');
  });

  it('reads meta.luck: missing or unreadable is a fair world, out of range is clamped', () => {
    expect(parseLuck(undefined)).toBe(0.5);
    expect(parseLuck('')).toBe(0.5);
    expect(parseLuck('lucky')).toBe(0.5);
    expect(parseLuck('0.2')).toBe(0.2);
    expect(parseLuck('7')).toBe(1);
    expect(parseLuck('-1')).toBe(0);
  });
});
