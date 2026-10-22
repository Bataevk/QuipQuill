import { useSyncExternalStore } from 'react';
import { useColorScheme as useRNColorScheme } from 'react-native';

import { getSchemeOverride, subscribeSchemeOverride } from '@/platform/web/color-scheme';

const subscribe = () => () => {};

/**
 * Static web rendering has no color scheme on the server, so report 'light'
 * until the client has hydrated. The player's choice in the profile (stage W.3)
 * wins over the system's.
 */
export function useColorScheme() {
  const hasHydrated = useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
  const override = useSyncExternalStore(subscribeSchemeOverride, getSchemeOverride, () => null);
  const colorScheme = useRNColorScheme();

  return hasHydrated ? (override ?? colorScheme) : 'light';
}
