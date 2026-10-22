/**
 * Asks before an action that cannot be undone, in the browser (stage W.3): react-native-web's
 * Alert does nothing, so the page's own dialog. It has no custom buttons: OK stands for `action`.
 */
export function confirmAction(title: string, message: string, _action: string, run: () => void): void {
  if (globalThis.confirm(`${title}\n\n${message}`)) run();
}
