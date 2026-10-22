/**
 * The app's fonts in the browser (stage W.3, ADR 69): the same files the expo-font plugin embeds
 * on the phone, as @font-face rules under the family names of `Fonts.default` (constants/theme),
 * with their real styles, so italics and the semibold are the drawn faces, not synthesized ones.
 * Loaded before the boot screen goes, so the first page does not jump from Georgia.
 */
import { Asset } from 'expo-asset';

const FACES: { family: string; style: 'normal' | 'italic'; weight: number; file: number }[] = [
  { family: 'IM Fell English', style: 'normal', weight: 400, file: require('@expo-google-fonts/im-fell-english/400Regular/IMFellEnglish_400Regular.ttf') },
  { family: 'IM Fell English', style: 'italic', weight: 400, file: require('@expo-google-fonts/im-fell-english/400Regular_Italic/IMFellEnglish_400Regular_Italic.ttf') },
  { family: 'Literata', style: 'normal', weight: 400, file: require('@expo-google-fonts/literata/400Regular/Literata_400Regular.ttf') },
  { family: 'Literata', style: 'italic', weight: 400, file: require('@expo-google-fonts/literata/400Regular_Italic/Literata_400Regular_Italic.ttf') },
  { family: 'Literata', style: 'normal', weight: 600, file: require('@expo-google-fonts/literata/600SemiBold/Literata_600SemiBold.ttf') },
];

/** Never holds the start longer than this: a slow network gets the fallback fonts first. */
const LOAD_TIMEOUT_MS = 3000;

let loaded: Promise<void> | undefined;

export function loadWebFonts(): Promise<void> {
  loaded ??= (async () => {
    if (typeof document === 'undefined' || typeof FontFace === 'undefined') return;
    const faces = FACES.map(
      ({ family, style, weight, file }) =>
        new FontFace(family, `url(${Asset.fromModule(file).uri})`, { style, weight: String(weight), display: 'swap' }),
    );
    for (const face of faces) document.fonts.add(face);
    await Promise.race([
      Promise.allSettled(faces.map((face) => face.load())),
      new Promise((resolve) => setTimeout(resolve, LOAD_TIMEOUT_MS)),
    ]);
  })();
  return loaded;
}
