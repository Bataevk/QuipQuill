// Must stay first: standard Web APIs the core expects (ADR 27).
import '@/platform/polyfills';

import { DarkTheme, DefaultTheme, Stack, ThemeProvider, router, useSegments } from 'expo-router';
import { useEffect } from 'react';
import { AppState, useColorScheme } from 'react-native';

import { getCloud } from '@/platform/services';

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

export default function RootLayout() {
  const colorScheme = useColorScheme();
  useAuthGate();
  return (
    <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
      <Stack screenOptions={{ headerBackTitle: 'Back' }}>
        <Stack.Screen name="index" options={{ title: 'foolore' }} />
        <Stack.Screen name="login" options={{ title: 'Sign in', headerBackVisible: false }} />
        <Stack.Screen name="stories" options={{ title: 'Stories' }} />
        <Stack.Screen name="characters/index" options={{ title: 'Characters' }} />
        <Stack.Screen name="characters/new" options={{ title: 'New character' }} />
        <Stack.Screen name="new-game" options={{ title: 'New game' }} />
        <Stack.Screen name="game/[id]" options={{ title: 'Game' }} />
      </Stack>
    </ThemeProvider>
  );
}
