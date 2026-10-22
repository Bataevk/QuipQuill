import { describe, expect, it } from 'vitest';

import { keyIntent } from './chat-keys';

describe('keyIntent', () => {
  it('sends on Enter with a mouse and a keyboard', () => {
    expect(keyIntent({ key: 'Enter' }, false)).toBe('send');
  });

  it('breaks the line on Shift+Enter', () => {
    expect(keyIntent({ key: 'Enter', shiftKey: true }, false)).toBeUndefined();
  });

  it('leaves Enter to the input method while it composes', () => {
    expect(keyIntent({ key: 'Enter', isComposing: true }, false)).toBeUndefined();
  });

  it('breaks the line on Enter on a touch screen', () => {
    expect(keyIntent({ key: 'Enter' }, true)).toBeUndefined();
  });

  it('ignores other keys', () => {
    expect(keyIntent({ key: 'a' }, false)).toBeUndefined();
  });
});
