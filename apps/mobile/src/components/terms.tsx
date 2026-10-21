import { ScrollView, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Button, Sheet } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { TERMS_VERSION } from '@/platform/onboarding';

/** The terms in three lines: the onboarding's step (D.6). */
export const TERMS_POINTS = [
  { title: 'You must be 16 or older.', text: '' },
  { title: 'What you write goes to our servers', text: 'and to the AI providers that write the story with you.' },
  { title: 'The AI can be wrong.', text: 'It is fiction, not advice.' },
];

/** A stub (ADR 60): the real terms and privacy policy come from a lawyer (backlog). */
const FULL_TEXT = `DRAFT — not the final legal text.

foolore is a prototype of an interactive story with an AI game master.

Who can play. You must be at least 16 years old.

What we keep. The app keeps your stories, heroes and games on this device. To play, it sends what you write and the parts of the story the master needs to our servers, which pass them to AI model providers to write the answer. We count how much each guest plays to keep the prototype within its limit. If you leave an email, we keep it only to tell you when foolore is released.

The AI. The stories are written by AI models. They can be wrong, strange or unsuitable; they are fiction, not advice.

Your rights. You can delete the app and its data at any time. Ask us to delete your email: write to the address on the store page.

Version ${TERMS_VERSION}.`;

/** The full terms in a sheet: from the onboarding and from the profile. */
export function TermsSheet({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  return (
    <Sheet visible={visible} onClose={onClose} title="Terms and privacy">
      <ScrollView>
        <View style={styles.full}>
          <ThemedText type="body">{FULL_TEXT}</ThemedText>
        </View>
      </ScrollView>
      <Button title="Close" onPress={onClose} />
    </Sheet>
  );
}

const styles = StyleSheet.create({
  full: { paddingBottom: Spacing.three },
});
