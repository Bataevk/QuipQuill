import { describe, expect, it } from 'vitest';

import { parseProse } from './prose';

const spans = (text: string, streaming = false) => parseProse(text, { streaming }).flatMap((b) => b.spans);

describe('parseProse', () => {
  it('keeps plain text as one paragraph', () => {
    expect(parseProse('The door creaks.')).toEqual([{ kind: 'paragraph', spans: [{ text: 'The door creaks.' }] }]);
  });

  it('splits paragraphs on blank lines and keeps single line breaks', () => {
    const blocks = parseProse('One.\nStill one.\n\n  \nTwo.');
    expect(blocks.map((b) => b.spans[0].text)).toEqual(['One.\nStill one.', 'Two.']);
  });

  it('reads italic, bold and the bold triple', () => {
    expect(spans('*She sets down the rag.* "Well?"')).toEqual([{ text: 'She sets down the rag.', italic: true }, { text: ' "Well?"' }]);
    expect(spans('It is **noon**.')).toEqual([{ text: 'It is ' }, { text: 'noon', bold: true }, { text: '.' }]);
    expect(spans('_quietly_ and ***now***')).toEqual([
      { text: 'quietly', italic: true },
      { text: ' and ' },
      { text: 'now', bold: true },
    ]);
  });

  it('nests italic in bold', () => {
    expect(spans('**a *b* c**')).toEqual([
      { text: 'a ', bold: true },
      { text: 'b', bold: true, italic: true },
      { text: ' c', bold: true },
    ]);
  });

  it('leaves stray and unflanked markers as text', () => {
    expect(spans('2 * 3 = 6')).toEqual([{ text: '2 * 3 = 6' }]);
    expect(spans('* a list item')).toEqual([{ text: '* a list item' }]);
    expect(spans('an *open marker')).toEqual([{ text: 'an *open marker' }]);
    expect(spans('snake_case_name')).toEqual([{ text: 'snake_case_name' }]);
    expect(spans('****')).toEqual([{ text: '****' }]);
  });

  it('honours escapes', () => {
    expect(spans('\\*not italic\\*')).toEqual([{ text: '*not italic*' }]);
  });

  it('reads a quote block', () => {
    expect(parseProse('She reads:\n\n> *To the bearer —*\n> come at noon.')).toEqual([
      { kind: 'paragraph', spans: [{ text: 'She reads:' }] },
      { kind: 'quote', spans: [{ text: 'To the bearer —', italic: true }, { text: '\ncome at noon.' }] },
    ]);
  });

  it('does not treat headings or lists as markup', () => {
    expect(spans('# Title\n- item')).toEqual([{ text: '# Title\n- item' }]);
  });

  describe('while streaming', () => {
    it('styles the rest after an open marker', () => {
      expect(spans('He says **noth', true)).toEqual([{ text: 'He says ' }, { text: 'noth', bold: true }]);
    });

    it('hides a marker run at the very end', () => {
      expect(spans('He says **nothing*', true)).toEqual([{ text: 'He says ' }, { text: 'nothing', bold: true }]);
      expect(spans('He says *', true)).toEqual([{ text: 'He says ' }]);
    });

    it('settles once the closer arrives', () => {
      expect(spans('He says **nothing**', true)).toEqual([{ text: 'He says ' }, { text: 'nothing', bold: true }]);
    });

    it('only the last paragraph is open', () => {
      expect(parseProse('an *open marker\n\nand *more', { streaming: true })).toEqual([
        { kind: 'paragraph', spans: [{ text: 'an *open marker' }] },
        { kind: 'paragraph', spans: [{ text: 'and ' }, { text: 'more', italic: true }] },
      ]);
    });
  });
});
