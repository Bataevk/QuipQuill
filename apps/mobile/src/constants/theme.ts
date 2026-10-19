/**
 * The design system's tokens (ADR 63): the light theme "Parchment" and the dark theme "Night",
 * fonts, spacing and radii. Screens take colours only from here.
 */

import { Platform } from 'react-native';

export const Colors = {
  light: {
    /** Page ground. */
    bg: '#F4ECDD',
    /** Cards, fields, sheets: one step above the ground. */
    surface: '#FBF6EC',
    /** Pressed and inset things: the player's lines, a pressed button. */
    sunken: '#EBE0CC',
    ink: '#2B2528',
    inkMuted: '#6E6259',
    line: '#D9CBB3',
    /** Crimson: the main action, a speaker's name, the selection. */
    accent: '#A3313A',
    onAccent: '#FBF6EC',
    /** Decoration only (bells, rules, frames), never text: 2.3:1 on parchment. */
    gold: '#C9963A',
    danger: '#B04A1F',
    scrim: 'rgba(43, 37, 40, 0.35)',
  },
  dark: {
    bg: '#1D1A1E',
    surface: '#27232A',
    sunken: '#161417',
    ink: '#EDE4D3',
    inkMuted: '#A89C8E',
    line: '#3A343C',
    accent: '#E06C70',
    onAccent: '#1D1A1E',
    gold: '#D9AE5C',
    danger: '#E0844F',
    scrim: 'rgba(0, 0, 0, 0.5)',
  },
} as const;

export type ThemeColor = keyof typeof Colors.light & keyof typeof Colors.dark;
export type Theme = { [K in ThemeColor]: string };

/**
 * Font families. The files are built into the app by the expo-font plugin (app.json), so a face
 * is named by its file on Android and by its PostScript name on iOS. Each face is its own family:
 * SDK 57 has no weight mapping for custom fonts, so never set `fontWeight` on these.
 * IM Fell English is for titles only (it gets muddy below ~20 pt); Literata is the story's text.
 */
export const Fonts = Platform.select({
  ios: {
    display: 'IM_FELL_English_Roman',
    displayItalic: 'IM_FELL_English_Italic',
    story: 'Literata-Regular',
    storyItalic: 'Literata-Italic',
    storyBold: 'Literata-SemiBold',
    sans: 'system-ui',
    mono: 'ui-monospace',
  },
  android: {
    display: 'IMFellEnglish_400Regular',
    displayItalic: 'IMFellEnglish_400Regular_Italic',
    story: 'Literata_400Regular',
    storyItalic: 'Literata_400Regular_Italic',
    storyBold: 'Literata_600SemiBold',
    sans: 'normal',
    mono: 'monospace',
  },
  default: {
    display: "'IM Fell English', Georgia, serif",
    displayItalic: "'IM Fell English', Georgia, serif",
    story: "Literata, Georgia, serif",
    storyItalic: "Literata, Georgia, serif",
    storyBold: "Literata, Georgia, serif",
    sans: "system-ui, -apple-system, 'Segoe UI', sans-serif",
    mono: 'ui-monospace, Menlo, Consolas, monospace',
  },
});

export const Spacing = {
  half: 2,
  one: 4,
  two: 8,
  three: 16,
  four: 24,
  five: 32,
  six: 64,
} as const;

export const Radius = {
  small: 8,
  medium: 12,
  large: 18,
  pill: 999,
} as const;

export const MaxContentWidth = 800;
