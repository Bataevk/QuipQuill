import type { Character } from '@/core/library/app-db';

/** "elf, female, 212 — A high elf archer…" */
export function characterLine(c: Pick<Character, 'race' | 'sex' | 'age' | 'description'>): string {
  const sheet = [c.race, c.sex, c.age ? `${c.age}` : ''].filter(Boolean).join(', ');
  return [sheet, c.description].filter(Boolean).join(' — ');
}
