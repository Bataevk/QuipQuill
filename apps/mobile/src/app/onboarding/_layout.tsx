import { Redirect, Stack } from 'expo-router';
import { useEffect, useState } from 'react';

import { isOnboarded } from '@/platform/onboarding';

/**
 * The first launch (stage D.6): its own stack without headers; each page draws its back link.
 * Once done it is never shown again: the browser's Back or a link into it (web, ADR 69) lands on
 * the main pages instead.
 */
export default function OnboardingLayout() {
  const done = useOnboarded();
  if (done === undefined) return null;
  if (done) return <Redirect href="/" />;
  return <Stack screenOptions={{ headerShown: false, gestureEnabled: true }} />;
}

/** Checked once, when the onboarding opens: finishing it here leads on to the game by itself. */
function useOnboarded(): boolean | undefined {
  const [done, setDone] = useState<boolean>();
  useEffect(() => {
    let active = true;
    void isOnboarded()
      .catch(() => false)
      .then((value) => active && setDone(value));
    return () => {
      active = false;
    };
  }, []);
  return done;
}
