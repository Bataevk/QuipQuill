import type { TextInputProps } from 'react-native';

/**
 * Platform parts of the chat field (stage T.1). On a phone the multiline field grows by itself up
 * to its `maxHeight`, and Enter breaks the line as in ChatGPT: the Send button sends. The web
 * version (`.web.ts`) sends on Enter with a keyboard and grows the textarea.
 */
export function useChatInputBehavior(_send: () => void, _maxHeight: number): Partial<TextInputProps> {
  return {};
}
