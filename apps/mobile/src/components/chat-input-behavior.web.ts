import type { TextInputProps } from 'react-native';

import { keyIntent } from '@/components/chat-keys';

/** A finger, not a mouse: Enter then breaks the line, as on the phone. */
const touchScreen = () => typeof window !== 'undefined' && window.matchMedia?.('(pointer: coarse)').matches === true;

/**
 * The web parts of the chat field (stage T.1): Enter sends and Shift+Enter breaks the line with a
 * keyboard (`keyIntent`); the textarea grows with its text up to `maxHeight`, then scrolls. A
 * textarea never shrinks by itself (its scroll height is at least its height), so the height is
 * reset before it is measured.
 */
export function useChatInputBehavior(send: () => void, maxHeight: number): Partial<TextInputProps> {
  return {
    numberOfLines: 1, // the textarea's rows (react-native-web): it starts at one line and grows
    onKeyPress: (event) => {
      const e = event as unknown as { key: string; shiftKey?: boolean; nativeEvent: { isComposing?: boolean }; preventDefault(): void };
      if (keyIntent({ key: e.key, shiftKey: e.shiftKey, isComposing: e.nativeEvent?.isComposing }, touchScreen()) !== 'send') return;
      e.preventDefault(); // react-native-web then neither inserts the line nor blurs
      send();
    },
    onChange: (event) => {
      const area = event.target as unknown as HTMLTextAreaElement | null;
      if (!area?.style) return;
      area.style.height = 'auto';
      area.style.height = `${Math.min(area.scrollHeight, maxHeight)}px`;
      area.style.overflowY = area.scrollHeight > maxHeight ? 'auto' : 'hidden';
    },
  };
}
