/**
 * Prose markup (ADR 67): the narrow piece of Markdown the master and the characters may use —
 * `*italic*` / `_italic_`, `**bold**` (`***x***` is bold too: there is no bold italic face),
 * paragraphs (a blank line) and `> quotes` for letters and inscriptions. Anything else (headings,
 * lists, code, links) stays plain text. Pure TypeScript: a renderer turns the blocks into views,
 * and replacing the parser with a library means only another `parseProse` with the same output.
 */

export interface ProseSpan {
  text: string;
  italic?: boolean;
  bold?: boolean;
}

export interface ProseBlock {
  kind: 'paragraph' | 'quote';
  /** Line breaks inside a block stay as `\n` in the spans. */
  spans: ProseSpan[];
}

export interface ProseOptions {
  /**
   * The text is still arriving: a marker left open at the end styles the rest as if it were
   * closed, and a marker run at the very end is hidden (it may be half of `**`).
   */
  streaming?: boolean;
}

export function parseProse(text: string, options: ProseOptions = {}): ProseBlock[] {
  const blocks: ProseBlock[] = [];
  const paragraphs = text.replace(/\r\n?/g, '\n').split(/\n(?:[ \t]*\n)+/);
  paragraphs.forEach((raw, i) => {
    const body = raw.replace(/^(?:[ \t]*\n)+|\s+$/g, '');
    if (!body) return;
    const lines = body.split('\n');
    const quote = lines.every((line) => /^\s*>/.test(line));
    const source = quote ? lines.map((line) => line.replace(/^\s*> ?/, '')).join('\n') : body;
    // Only the last paragraph is still being written.
    const streaming = !!options.streaming && i === paragraphs.length - 1;
    const spans = parseInline(source, streaming);
    if (spans.length) blocks.push({ kind: quote ? 'quote' : 'paragraph', spans });
  });
  return blocks;
}

type Strength = 'italic' | 'bold';

interface Delimiter {
  kind: 'delim';
  char: '*' | '_';
  length: number;
  strength: Strength | null;
  canOpen: boolean;
  canClose: boolean;
  /** -1: plain text; otherwise the index of the token it pairs with. */
  pair: number;
  /** Opened but never closed while streaming: styles everything after it. */
  openToEnd?: boolean;
}
type Token = { kind: 'text'; text: string } | Delimiter;

const isSpace = (c: string | undefined) => c === undefined || /\s/.test(c);
const isWord = (c: string | undefined) => c !== undefined && /[\p{L}\p{N}]/u.test(c);
const isPunct = (c: string | undefined) => c !== undefined && /[\p{P}\p{S}]/u.test(c);

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let text = '';
  const flush = () => {
    if (text) tokens.push({ kind: 'text', text });
    text = '';
  };
  for (let i = 0; i < source.length; ) {
    const c = source[i];
    if (c === '\\' && /[*_\\>]/.test(source[i + 1] ?? '')) {
      text += source[i + 1];
      i += 2;
      continue;
    }
    if (c !== '*' && c !== '_') {
      text += c;
      i += 1;
      continue;
    }
    let j = i;
    while (source[j] === c) j += 1;
    const length = j - i;
    const before = source[i - 1];
    const after = source[j];
    // Flanking as in CommonMark, simplified: a marker opens before a non-space and closes after one.
    let canOpen = !isSpace(after) && !(isPunct(after) && !isSpace(before) && !isPunct(before));
    let canClose = !isSpace(before) && !(isPunct(before) && !isSpace(after) && !isPunct(after));
    // `_` never works inside a word (snake_case, names like Mrs_Hale).
    if (c === '_') {
      canOpen &&= !isWord(before);
      canClose &&= !isWord(after);
    }
    flush();
    tokens.push({
      kind: 'delim',
      char: c,
      length,
      strength: length === 1 ? 'italic' : length <= 3 ? 'bold' : null,
      canOpen,
      canClose,
      pair: -1,
    });
    i = j;
  }
  flush();
  return tokens;
}

function parseInline(source: string, streaming: boolean): ProseSpan[] {
  const tokens = tokenize(source);

  // Pair closers with the nearest open marker of the same character and strength.
  const open: number[] = [];
  tokens.forEach((token, i) => {
    if (token.kind !== 'delim' || !token.strength) return;
    if (token.canClose) {
      const at = open.findLastIndex((o) => {
        const opener = tokens[o] as Delimiter;
        return opener.char === token.char && opener.strength === token.strength;
      });
      if (at >= 0) {
        const o = open[at];
        (tokens[o] as Delimiter).pair = i;
        token.pair = o;
        open.splice(at); // markers opened inside the pair and left open are plain text
        return;
      }
    }
    if (token.canOpen) open.push(i);
  });

  if (streaming) {
    // A run at the very end may be the start of a closer still arriving: hide it.
    const last = tokens.at(-1);
    if (last?.kind === 'delim' && last.pair < 0) {
      tokens.pop();
      const at = open.indexOf(tokens.length);
      if (at >= 0) open.splice(at, 1);
    }
    for (const o of open) (tokens[o] as Delimiter).openToEnd = true;
  }

  const spans: ProseSpan[] = [];
  let italic = 0;
  let bold = 0;
  const push = (text: string) => {
    if (!text) return;
    const style = { ...(italic > 0 ? { italic: true } : {}), ...(bold > 0 ? { bold: true } : {}) };
    const prev = spans.at(-1);
    if (prev && !!prev.italic === !!style.italic && !!prev.bold === !!style.bold) prev.text += text;
    else spans.push({ text, ...style });
  };
  tokens.forEach((token, i) => {
    if (token.kind === 'text') return push(token.text);
    const paired = token.pair >= 0 || token.openToEnd;
    if (!paired) return push(token.char.repeat(token.length));
    const opening = token.openToEnd || token.pair > i;
    const step = opening ? 1 : -1;
    if (token.strength === 'italic') italic += step;
    else bold += step;
  });
  return spans;
}
