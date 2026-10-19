import { useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Button, Chip, ErrorText, Field } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import type { Character } from '@/core/library/app-db';

export type HeroSheet = Pick<Character, 'name' | 'sex' | 'age' | 'race' | 'description'>;

const SEXES = ['Woman', 'Man', 'Non-binary'];
const ORIGINS = ['Human', 'Elf', 'Half-elf', 'Dwarf', 'Halfling', 'Automaton'];

/**
 * The hero's sheet on one screen (stage D.7): name, who, from where, a few lines about them.
 * The step-by-step version for the first launch comes with onboarding (D.6).
 */
export function HeroForm({
  initial,
  submitTitle,
  onSubmit,
}: {
  initial?: HeroSheet;
  submitTitle: string;
  onSubmit: (sheet: HeroSheet) => Promise<void>;
}) {
  const [name, setName] = useState(initial?.name ?? '');
  const [sex, setSex] = useState(initial?.sex ?? '');
  const [age, setAge] = useState(initial?.age ? String(initial.age) : '');
  const [race, setRace] = useState(initial?.race ?? '');
  const [raceKey, setRaceKey] = useState(0); // a new key reseeds the (uncontrolled) field from a chip
  const [description, setDescription] = useState(initial?.description ?? '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const pickOrigin = (origin: string) => {
    setRace(origin);
    setRaceKey((k) => k + 1);
  };

  const submit = async () => {
    const years = age.trim() ? Number(age) : null;
    if (years !== null && (!Number.isInteger(years) || years <= 0)) return setError('Age must be a whole number.');
    setBusy(true);
    setError('');
    try {
      await onSubmit({
        name: name.trim(),
        sex: sex.trim() || null,
        age: years,
        race: race.trim() || null,
        description: description.trim(),
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <View style={styles.form}>
      <Field label="Name" value={name} onChangeText={setName} autoCorrect={false} placeholder="What do they call you?" />

      <View style={styles.group}>
        <ThemedText type="caption" themeColor="inkMuted">
          Who are you
        </ThemedText>
        <View style={styles.chips}>
          {SEXES.map((s) => (
            <Chip key={s} label={s} selected={sex === s} onPress={() => setSex(sex === s ? '' : s)} />
          ))}
        </View>
      </View>

      <Field label="Age" value={age} onChangeText={setAge} keyboardType="number-pad" placeholder="Leave empty if it does not matter" />

      <View style={styles.group}>
        <Field key={raceKey} label="Where you are from" value={race} onChangeText={setRace} autoCorrect={false} placeholder="A people, a land, a trade" />
        <View style={styles.chips}>
          {ORIGINS.map((o) => (
            <Chip key={o} label={o} selected={race === o} onPress={() => pickOrigin(o)} />
          ))}
        </View>
      </View>

      <Field
        label="A few lines about you"
        value={description}
        onChangeText={setDescription}
        multiline
        placeholder="A locksmith who owes the wrong people. Quick hands, quicker tongue."
        style={styles.description}
      />
      <ThemedText type="caption" themeColor="inkMuted">
        Each story fits your hero into its world: a dwarf in a city of steam may become a mine-born engineer.
      </ThemedText>

      <Button title={submitTitle} variant="primary" onPress={submit} busy={busy} disabled={!name.trim()} />
      <ErrorText>{error}</ErrorText>
    </View>
  );
}

const styles = StyleSheet.create({
  form: { gap: Spacing.three },
  group: { gap: Spacing.two },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.two },
  description: { minHeight: 112, textAlignVertical: 'top' },
});
