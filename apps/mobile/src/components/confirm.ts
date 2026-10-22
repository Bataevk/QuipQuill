import { Alert } from 'react-native';

/** Asks before an action that cannot be undone; `run` only on yes. The browser's: confirm.web.ts. */
export function confirmAction(title: string, message: string, action: string, run: () => void): void {
  Alert.alert(title, message, [
    { text: 'Cancel', style: 'cancel' },
    { text: action, style: 'destructive', onPress: run },
  ]);
}
