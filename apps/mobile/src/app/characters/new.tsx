import { router } from 'expo-router';
import { useState } from 'react';

import { Button, ErrorText, Field, Screen, errorText } from '@/components/ui';
import { newId, nowIso } from '@/core/util/ids';
import { getLibrary } from '@/platform/services';

/** The character sheet: name, sex, age, race, description (agent.md#старт-сессии). */
export default function NewCharacterScreen() {
  const [name, setName] = useState('');
  const [sex, setSex] = useState('');
  const [age, setAge] = useState('');
  const [race, setRace] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState('');

  const save = async () => {
    const years = age.trim() ? Number(age) : null;
    if (years !== null && (!Number.isInteger(years) || years <= 0)) return setError('Age must be a whole number.');
    try {
      await (await getLibrary()).repository.addCharacter({
        id: newId(),
        name: name.trim(),
        sex: sex.trim() || null,
        age: years,
        race: race.trim() || null,
        description: description.trim(),
        createdAt: nowIso(),
      });
      router.back();
    } catch (e) {
      setError(errorText(e));
    }
  };

  return (
    <Screen>
      <Field label="Name" value={name} onChangeText={setName} autoCorrect={false} />
      <Field label="Sex" value={sex} onChangeText={setSex} autoCorrect={false} />
      <Field label="Age" value={age} onChangeText={setAge} keyboardType="number-pad" />
      <Field label="Race" value={race} onChangeText={setRace} autoCorrect={false} />
      <Field label="Description" value={description} onChangeText={setDescription} multiline style={{ minHeight: 96 }} />
      <Button title="Save" onPress={save} disabled={!name.trim()} />
      <ErrorText>{error}</ErrorText>
    </Screen>
  );
}
