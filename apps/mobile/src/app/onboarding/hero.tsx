import { router } from 'expo-router';

import { useHeroWizard } from '@/components/hero-wizard';
import { OnboardingPage } from '@/components/onboarding';
import { newId, nowIso } from '@/core/util/ids';
import { getLibrary } from '@/platform/services';

/** Your hero (D.6): the step-by-step wizard; "Back" at the first question returns to the terms. */
export default function HeroScreen() {
  const wizard = useHeroWizard({
    createTitle: 'This is me',
    onCreate: async (sheet) => {
      const id = newId();
      await (await getLibrary()).repository.addCharacter({ id, ...sheet, createdAt: nowIso() });
      router.push({ pathname: '/onboarding/notify', params: { hero: id } });
    },
  });
  return (
    <OnboardingPage step={1} onBack={wizard.atStart ? () => router.back() : wizard.back} footer={wizard.footer}>
      {wizard.body}
    </OnboardingPage>
  );
}
