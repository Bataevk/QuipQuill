/**
 * What a key does in the chat field (stage T.1), as in Claude and ChatGPT: with a mouse and a
 * keyboard, Enter sends and Shift+Enter breaks the line; on a touch screen Enter breaks the line
 * and the Send button sends. No React Native here, so it is tested in Node.
 */
export interface ChatKey {
  key: string;
  shiftKey?: boolean;
  /** An input method (IME) is composing: its Enter picks the word, not the move. */
  isComposing?: boolean;
}

/** `send` — the move goes; `undefined` — the field does what it always does (a new line). */
export function keyIntent(event: ChatKey, touch: boolean): 'send' | undefined {
  if (touch || event.key !== 'Enter' || event.shiftKey || event.isComposing) return undefined;
  return 'send';
}
