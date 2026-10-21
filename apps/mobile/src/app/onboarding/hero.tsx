import { router } from 'expo-router';
import { useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { sheetLine } from '@/components/format';
import { OnboardingPage } from '@/components/onboarding';
import { ThemedText } from '@/components/themed-text';
import { Button, Card, Chip, ErrorText, Field } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { newId, nowIso } from '@/core/util/ids';
import { getLibrary } from '@/platform/services';

const SEXES = ['Woman', 'Man', 'Non-binary'];
const AGES = [
  { label: 'Young', age: 19 },
  { label: 'Grown', age: 32 },
  { label: 'Seasoned', age: 47 },
  { label: 'Old', age: 68 },
];
const ORIGINS = ['Human', 'Elf', 'Half-elf', 'Dwarf', 'Halfling', 'Automaton'];
const EXAMPLES = [
  'A locksmith who owes the wrong people. Quick hands, quicker tongue.',
  'A retired soldier who keeps a diary of every lie she has told.',
  'A travelling actor, charming and broke, who can play anyone but himself.',
];

/** The hero's steps: name, who, from where, about, then the card. */
const STEPS = ['name', 'who', 'origin', 'about', 'card'] as const;

/**
 * Your hero, step by step (D.6): one question a screen, chips for the quick answers, no AI.
 * "Back" goes to the previous question and keeps every answer.
 */
export default function HeroScreen() {
  const [step, setStep] = useState(0);
  const [name, setName] = useState('');
  const [sex, setSex] = useState('');
  const [age, setAge] = useState<number | null>(null);
  const [race, setRace] = useState('');
  const [raceKey, setRaceKey] = useState(0); // a new key reseeds the (uncontrolled) field from a chip
  const [description, setDescription] = useState('');
  const [descriptionKey, setDescriptionKey] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const at = STEPS[step];
  const back = () => (step === 0 ? router.back() : setStep(step - 1));
  const next = () => setStep(step + 1);
  const canGoOn = at === 'name' ? !!name.trim() : at === 'about' ? !!description.trim() : true;

  const create = async () => {
    setBusy(true);
    setError('');
    try {
      const id = newId();
      await (await getLibrary()).repository.addCharacter({
        id,
        name: name.trim(),
        sex: sex || null,
        age,
        race: race.trim() || null,
        description: description.trim(),
        createdAt: nowIso(),
      });
      router.push({ pathname: '/onboarding/notify', params: { hero: id } });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const footer =
    at === 'card' ? (
      <>
        <Button title="Change something" variant="ghost" onPress={() => setStep(0)} />
        <Button title="This is me" variant="primary" busy={busy} onPress={() => void create()} />
      </>
    ) : (
      <Button title="Next" variant="primary" disabled={!canGoOn} onPress={next} />
    );

  return (
    <OnboardingPage step={1} onBack={back} footer={footer}>
      <ThemedText type="caption" themeColor="inkMuted" style={styles.eyebrow}>
        YOUR HERO · {Math.min(step + 1, 4)} OF 4
      </ThemedText>

      {at === 'name' ? (
        <>
          <ThemedText type="title">What do they call you?</ThemedText>
          <Field key="name" value={name} onChangeText={setName} placeholder="A name" autoCorrect={false} autoFocus returnKeyType="next" onSubmitEditing={() => name.trim() && next()} />
        </>
      ) : null}

      {at === 'who' ? (
        <>
          <ThemedText type="title">Who are you?</ThemedText>
          <View style={styles.chips}>
            {SEXES.map((s) => (
              <Chip key={s} label={s} selected={sex === s} onPress={() => setSex(sex === s ? '' : s)} />
            ))}
          </View>
          <ThemedText type="caption" themeColor="inkMuted">
            How old
          </ThemedText>
          <View style={styles.chips}>
            {AGES.map((a) => (
              <Chip key={a.label} label={a.label} selected={age === a.age} onPress={() => setAge(age === a.age ? null : a.age)} />
            ))}
          </View>
          <ThemedText type="caption" themeColor="inkMuted">
            Both can stay empty: the story will decide.
          </ThemedText>
        </>
      ) : null}

      {at === 'origin' ? (
        <>
          <ThemedText type="title">Where are you from?</ThemedText>
          <View style={styles.chips}>
            {ORIGINS.map((o) => (
              <Chip
                key={o}
                label={o}
                selected={race === o}
                onPress={() => {
                  setRace(o);
                  setRaceKey((k) => k + 1);
                }}
              />
            ))}
          </View>
          <Field key={`race-${raceKey}`} value={race} onChangeText={setRace} placeholder="Or a people, a land, a trade of your own" autoCorrect={false} />
          <ThemedText type="caption" themeColor="inkMuted">
            Each story fits your hero into its world: a dwarf in a city of steam may become a mine-born engineer.
          </ThemedText>
        </>
      ) : null}

      {at === 'about' ? (
        <>
          <ThemedText type="title">A few lines about you</ThemedText>
          <Field
            key={`about-${descriptionKey}`}
            value={description}
            onChangeText={setDescription}
            multiline
            placeholder="What you do, what you want, what you carry"
            style={styles.about}
          />
          <ThemedText type="caption" themeColor="inkMuted">
            For example — tap one to start from it:
          </ThemedText>
          {EXAMPLES.map((example) => (
            <Button
              key={example}
              title={example}
              variant="secondary"
              onPress={() => {
                setDescription(example);
                setDescriptionKey((k) => k + 1);
              }}
            />
          ))}
        </>
      ) : null}

      {at === 'card' ? (
        <>
          <ThemedText type="title">This is you</ThemedText>
          <Card>
            <ThemedText type="title">{name.trim()}</ThemedText>
            {sheetLine({ race: race.trim() || null, sex: sex || null, age }) ? (
              <ThemedText type="caption" themeColor="inkMuted">
                {sheetLine({ race: race.trim() || null, sex: sex || null, age })}
              </ThemedText>
            ) : null}
            <ThemedText type="story">{description.trim()}</ThemedText>
          </Card>
          <ErrorText>{error}</ErrorText>
        </>
      ) : null}
    </OnboardingPage>
  );
}

const styles = StyleSheet.create({
  eyebrow: { letterSpacing: 1.2 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.two },
  about: { minHeight: 112, textAlignVertical: 'top' },
});
