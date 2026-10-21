// Must stay first: standard Web APIs the core expects (ADR 27).
import '@/platform/polyfills';

import { DarkTheme, DefaultTheme, Stack, ThemeProvider, router, type Theme } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { useEffect, useMemo, useState } from 'react';
import { AppState } from 'react-native';

import { BootScreen } from '@/components/boot-screen';
import { Colors, Fonts } from '@/constants/theme';
import { useScheme } from '@/hooks/use-theme';
import { ensureSignedIn, getCloud, getLibrary } from '@/platform/services';
import { ensureBundledStory } from '@/platform/bundled-story';
import { isOnboarded } from '@/platform/onboarding';
import { restoreThemePreference } from '@/platform/theme-preference';

// The native splash stays until the boot screen takes over (stage D.3). The player's theme and
// the library open behind it; the built-in story imports in the background (the Stories tab
// waits for it and shows an error if it fails).
void SplashScreen.preventAutoHideAsync();
// At least one swing of the bells out and back, so the start reads as a moment, not a flicker.
const MIN_BOOT_MS = 1200;
// Whether the first launch is done (stage D.6): the boot screen hides the jump to onboarding.
const onboarded = isOnboarded().catch(() => true);
const booting = Promise.allSettled([
  restoreThemePreference(),
  getLibrary(),
  getCloud().auth.isSignedIn(),
  onboarded,
  new Promise((resolve) => setTimeout(resolve, MIN_BOOT_MS)),
]);
void ensureBundledStory().catch(() => {});

/**
 * True once the app's data is open: the boot screen fades away. A first launch goes to the
 * onboarding (stage D.6) behind the boot screen, so the main pages never flash.
 */
function useBooted(): boolean {
  const [booted, setBooted] = useState(false);
  useEffect(() => {
    let active = true;
    void booting
      .then(() => onboarded)
      .then((done) => {
        if (!active) return;
        if (!done) router.replace('/onboarding');
        setBooted(true);
      });
    return () => {
      active = false;
    };
  }, []);
  return booted;
}

/**
 * No sign-in in the prototype (ADR 66): without tokens the device becomes a guest of the cloud and
 * the socket starts; when the cloud drops the session (`signed-out`), a new guest takes over.
 * Signing in with an invite is in the profile (`/login`).
 */
function useCloudSession() {
  useEffect(() => {
    void ensureSignedIn().catch(() => {}); // offline: the next action tries again
    const off = getCloud().socket.onStatus((status) => {
      if (status === 'signed-out') void ensureSignedIn().catch(() => {});
    });
    return off;
  }, []);

  // Back from the background: the OS may have killed the socket while the app was suspended,
  // often without a close event (stage 6.2e); reconnect at once and resume (the server replays
  // a generation in flight or repeats its `done`).
  // Only after real backgrounding: `inactive` alone (control centre, app switcher, system
  // prompts) keeps the socket alive, and a needless reconnect would replay the stream for
  // nothing. "Was in the background" survives an `inactive` on the way back.
  useEffect(() => {
    let away = AppState.currentState === 'background';
    const sub = AppState.addEventListener('change', (state) => {
      if (state === 'background') away = true;
      if (state !== 'active' || !away) return;
      away = false;
      const { socket } = getCloud();
      if (socket.status === 'ready' || socket.status === 'reconnecting' || socket.status === 'connecting') {
        socket.reconnectNow();
      }
    });
    return () => sub.remove();
  }, []);
}

/** React Navigation's theme from the design tokens (ADR 63): headers and backgrounds match the screens. */
function useNavigationTheme(): Theme {
  const scheme = useScheme();
  return useMemo(() => {
    const base = scheme === 'dark' ? DarkTheme : DefaultTheme;
    const c = Colors[scheme];
    return {
      ...base,
      colors: { ...base.colors, primary: c.accent, background: c.bg, card: c.bg, text: c.ink, border: c.line, notification: c.accent },
    };
  }, [scheme]);
}

export default function RootLayout() {
  const theme = useNavigationTheme();
  const booted = useBooted();
  useCloudSession();
  return (
    <ThemeProvider value={theme}>
      <Stack
        screenOptions={{
          headerBackTitle: 'Back',
          headerShadowVisible: false,
          headerTitleStyle: { fontFamily: Fonts.display, fontSize: 22 },
        }}>
        <Stack.Screen name="(tabs)" options={{ headerShown: false, title: 'foolore' }} />
        <Stack.Screen name="onboarding" options={{ headerShown: false, gestureEnabled: false }} />
        <Stack.Screen name="login" options={{ title: 'Sign in' }} />
        <Stack.Screen name="story/[id]" options={{ title: '' }} />
        <Stack.Screen name="heroes/new" options={{ title: 'New hero' }} />
        <Stack.Screen name="heroes/[id]" options={{ title: 'Hero' }} />
        <Stack.Screen name="game/[id]" options={{ title: '' }} />
      </Stack>
      <BootScreen ready={booted} />
    </ThemeProvider>
  );
}
