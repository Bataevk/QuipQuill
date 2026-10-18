/**
 * Name matching shared by the world queries and tools: the model may say "the brass lantern",
 * "Brass Lantern" or "lantern". Mirrors builder's normalization (builder/…/text.py).
 */
const ARTICLES = /^(the|a|an)\s+/;

export function normalizeName(name: string): string {
  return name
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^\p{L}\p{N}\s-]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(ARTICLES, '');
}

/** "The Brass Lantern" → "brass-lantern" (ids: `<prefix>:<slug>`, as builder does). */
export function slugify(name: string): string {
  const slug = normalizeName(name)
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug || 'entity';
}

/** Similarity in [0, 1] from Levenshtein distance (same idea as rapidfuzz `ratio`). */
export function similarity(a: string, b: string): number {
  if (a === b) return 1;
  if (!a.length || !b.length) return 0;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    for (let j = 1; j <= b.length; j++) {
      row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = row;
  }
  return 1 - prev[b.length] / Math.max(a.length, b.length);
}

/**
 * Names by entity id for one piece of text the model reads, each looked up once; an id the world
 * does not know stays as it is. Short-lived: a rename later is not seen.
 */
type Named = { id: string; name: string };

export class NameBook {
  readonly known = new Map<string, string>();
  private readonly lookup: (id: string) => Promise<Named | undefined>;

  constructor(lookup: (id: string) => Promise<Named | undefined>, entities: Named[] = []) {
    this.lookup = lookup;
    this.remember(entities);
  }

  remember(entities: Named[]): void {
    for (const e of entities) this.known.set(e.id, e.name);
  }

  async of(id: string): Promise<string> {
    if (!this.known.has(id)) this.known.set(id, (await this.lookup(id))?.name ?? id);
    return this.known.get(id)!;
  }
}
