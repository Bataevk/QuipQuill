// Must stay first: standard Web APIs the core expects (ADR 27).
import '@/platform/polyfills';

import { DarkTheme, DefaultTheme, Stack, ThemeProvider, router, useSegments, type Theme } from 'expo-router';
import { useEffect, useMemo } from 'react';
import { AppState } from 'react-native';

import { Colors, Fonts } from '@/constants/theme';
import { useScheme } from '@/hooks/use-theme';
import { getCloud } from '@/platform/services';
import { restoreThemePreference } from '@/platform/theme-preference';

// The player's theme (Profile) before the first screen settles.
void restoreThemePreference().catch(() => {});

/** Signed out → the login screen; signed in → the socket runs (stage 5.1). */
function useAuthGate() {
  const segments = useSegments();
  const onLogin = segments[0] === 'login';

  useEffect(() => {
    const { auth, socket } = getCloud();
    let active = true;
    void auth.isSignedIn().then((signedIn) => {
      if (!active) return;
      if (signedIn) socket.start();
      else if (!onLogin) router.replace('/login');
    });
    const off = socket.onStatus((status) => {
      if (status === 'signed-out') router.replace('/login');
    });
    return () => {
      active = false;
      off();
    };
  }, [onLogin]);

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
  useAuthGate();
  return (
    <ThemeProvider value={theme}>
      <Stack
        screenOptions={{
          headerBackTitle: 'Back',
          headerShadowVisible: false,
          headerTitleStyle: { fontFamily: Fonts.display, fontSize: 22 },
        }}>
        <Stack.Screen name="(tabs)" options={{ headerShown: false, title: 'foolore' }} />
        <Stack.Screen name="login" options={{ title: 'Sign in', headerBackVisible: false }} />
        <Stack.Screen name="story/[id]" options={{ title: '' }} />
        <Stack.Screen name="heroes/new" options={{ title: 'New hero' }} />
        <Stack.Screen name="heroes/[id]" options={{ title: 'Hero' }} />
        <Stack.Screen name="game/[id]" options={{ title: '' }} />
      </Stack>
    </ThemeProvider>
  );
}
