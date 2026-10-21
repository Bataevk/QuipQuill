import { Stack } from 'expo-router';

/** The first launch (stage D.6): its own stack without headers; each page draws its back link. */
export default function OnboardingLayout() {
  return <Stack screenOptions={{ headerShown: false, gestureEnabled: true }} />;
}
