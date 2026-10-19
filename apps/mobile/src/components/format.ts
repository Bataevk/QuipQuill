import type { Character } from '@/core/library/app-db';

/** "elf, female, 212 — A high elf archer…" */
export function characterLine(c: Pick<Character, 'race' | 'sex' | 'age' | 'description'>): string {
  const sheet = sheetLine(c);
  return [sheet, c.description].filter(Boolean).join(' — ');
}

/** "elf, female, 212" — the short part of the sheet. */
export function sheetLine(c: Pick<Character, 'race' | 'sex' | 'age'>): string {
  return [c.race, c.sex, c.age ? `${c.age}` : ''].filter(Boolean).join(', ');
}

/** "15 Oct, 19:31" in the device's time zone (records keep UTC ISO-8601). */
export function localTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
}
