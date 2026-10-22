import { useRef, useState } from 'react';
import { StyleSheet, TextInput, View } from 'react-native';

import { useChatInputBehavior } from '@/components/chat-input-behavior';
import { Button } from '@/components/ui';
import { Radius, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

const LINE = 22;
/** The field grows up to six lines, then scrolls inside. */
const MAX_HEIGHT = LINE * 6 + Spacing.three * 2;

/**
 * The player's field in the game (stage T.1), as in Claude and ChatGPT: it grows with the text,
 * and a long move fits on several lines. Enter: `useChatInputBehavior` (a new line on a phone,
 * sending with a keyboard on the web). Uncontrolled, like `Field` (ui.tsx): a controlled
 * TextInput loses characters while the player types; a new `key` clears it.
 */
export function ChatInput({
  placeholder,
  editable,
  onSend,
}: {
  placeholder: string;
  editable: boolean;
  onSend: (text: string) => void;
}) {
  const theme = useTheme();
  const [text, setText] = useState('');
  const [inputKey, setInputKey] = useState(0);
  const [focused, setFocused] = useState(false);
  const current = useRef('');
  const send = () => {
    const value = current.current.trim();
    if (!value || !editable) return;
    current.current = '';
    setText('');
    setInputKey((k) => k + 1);
    onSend(value);
  };
  const behavior = useChatInputBehavior(send, MAX_HEIGHT);
  return (
    <View style={styles.row}>
      <TextInput
        key={inputKey}
        multiline
        editable={editable}
        placeholder={placeholder}
        placeholderTextColor={theme.inkMuted}
        selectionColor={theme.accent}
        cursorColor={theme.accent}
        textAlignVertical="top"
        submitBehavior="newline"
        onChangeText={(value) => {
          current.current = value;
          setText(value);
        }}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        {...behavior}
        style={[
          styles.input,
          { color: theme.ink, backgroundColor: theme.surface, borderColor: focused ? theme.accent : theme.line },
        ]}
      />
      <Button title="Send" variant="primary" onPress={send} disabled={!editable || !text.trim()} />
    </View>
  );
}

const styles = StyleSheet.create({
  row: { alignItems: 'flex-end', flexDirection: 'row', gap: Spacing.two },
  input: {
    borderRadius: Radius.medium,
    borderWidth: StyleSheet.hairlineWidth * 2,
    flex: 1,
    fontSize: 16,
    lineHeight: LINE,
    maxHeight: MAX_HEIGHT,
    minHeight: 48,
    paddingHorizontal: Spacing.three,
    paddingVertical: Spacing.two + Spacing.one,
  },
});
