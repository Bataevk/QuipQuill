import { router, useLocalSearchParams } from 'expo-router';

import { OnboardingPage } from '@/components/onboarding';
import { ThemedText } from '@/components/themed-text';
import { Button } from '@/components/ui';
import { WaitlistForm } from '@/components/waitlist-form';

/** Release news (D.6, ADR 66): an email for the waitlist if the player wants it; "Skip" is fine. */
export default function NotifyScreen() {
  const { hero } = useLocalSearchParams<{ hero: string }>();
  const next = () => router.push({ pathname: '/onboarding/story', params: { hero } });
  return (
    <OnboardingPage step={2} footer={<Button title="Continue" variant="primary" onPress={next} />}>
      <ThemedText type="title">This is a prototype</ThemedText>
      <ThemedText type="story">
        You can play about thirty turns here. If you would like to hear when the full version of foolore is out, leave
        an email — or just go on.
      </ThemedText>
      <WaitlistForm source="onboarding" />
    </OnboardingPage>
  );
}
