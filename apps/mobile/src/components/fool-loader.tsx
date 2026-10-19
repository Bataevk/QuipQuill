/**
 * «Бубенцы» — the loader (stage D.3, ADR 63). The emblem's bells swing like pendulums about the
 * hat's tips, the sun and the moon in counter-phase; the sun's rays turn slowly; the emblem
 * breathes. With Reduce Motion nothing swings — the bells only fade softly in and out.
 * The art is the emblem itself (icons/render.mjs → assets/images/loader-*), so the first frame
 * is the splash screen's.
 */
import { useLayoutEffect } from 'react';
import { Image, StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';
import Animated, {
  Easing,
  cancelAnimation,
  useAnimatedStyle,
  useReducedMotion,
  useSharedValue,
  withRepeat,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';

import { ThemedText } from '@/components/themed-text';
import { Radius, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

const EMBLEM = require('@/assets/images/loader-emblem.png');
const SUN_RAYS = require('@/assets/images/loader-sun-rays.png');
const SUN_BODY = require('@/assets/images/loader-sun-body.png');
const MOON = require('@/assets/images/loader-moon.png');

/** Half a swing (one side to the other), in ms; a full pendulum period is twice this. */
const SWING_MS = 800;
/** How far a bell swings each way, in degrees. */
const SWING_DEG = 18;
/** One turn of the sun's rays. */
const RAYS_MS = 14_000;
/** The emblem's breath: in and out. */
const BREATH_MS = 2400;

/**
 * Where the layers sit on the emblem, as fractions of its side (the emblem's viewBox is 900
 * units from 62; the art is scaled 0.95 about (512, 520) — icons/foolore.mjs).
 */
const at = (x: number, y: number) => ({ left: (512 + 0.95 * (x - 512) - 62) / 900, top: (520 + 0.95 * (y - 512) - 62) / 900 });
const span = (units: number) => (units * 0.95) / 900;
const SUN_FRAME = { ...at(160, 390), size: span(124) };
const MOON_FRAME = { ...at(711, 428), size: span(180) };
/** The sun hangs from its tip: 29 of its frame's 124 units from the top; the moon's tip is its frame's centre. */
const SUN_PIVOT = '50% 23.4%';
const MOON_PIVOT = '50% 50%';

/**
 * A pendulum 0 → 1 → 0, eased like a real swing, forever; held at 0.5 (at rest) with Reduce Motion.
 * Started in a layout effect: at app start the JS thread is busy, and passive effects would leave
 * the bells still for seconds (seen on the simulator).
 */
function usePendulum(still: boolean): SharedValue<number> {
  const t = useSharedValue(0.5);
  useLayoutEffect(() => {
    if (still) {
      t.value = 0.5;
      return;
    }
    t.value = 0.5;
    // From rest to one side first (half the time), then side to side.
    t.value = withTiming(1, { duration: SWING_MS / 2, easing: Easing.out(Easing.sin) }, () => {
      t.value = withRepeat(withTiming(0, { duration: SWING_MS, easing: Easing.inOut(Easing.sin) }), -1, true);
    });
    return () => cancelAnimation(t);
  }, [still, t]);
  return t;
}

function useLoop(still: boolean, duration: number, reverse: boolean, easing = Easing.inOut(Easing.sin)): SharedValue<number> {
  const t = useSharedValue(0);
  useLayoutEffect(() => {
    t.value = 0;
    if (still) return;
    t.value = withRepeat(withTiming(1, { duration, easing }), -1, reverse);
    return () => cancelAnimation(t);
  }, [still, duration, reverse, easing, t]);
  return t;
}

/** The bells' motion, shared by the emblem and the inline loader. */
function useBells() {
  const still = useReducedMotion();
  const swing = usePendulum(still);
  const rays = useLoop(still, RAYS_MS, false, Easing.linear);
  // Reduce Motion: a soft fade instead of the swing (the only thing that moves).
  const fade = useLoop(!still, 1200, true);

  const sun = useAnimatedStyle(() => ({
    opacity: still ? 1 - 0.4 * fade.value : 1,
    transform: [{ rotate: `${(swing.value * 2 - 1) * SWING_DEG}deg` }],
  }));
  const moon = useAnimatedStyle(() => ({
    opacity: still ? 0.6 + 0.4 * fade.value : 1,
    transform: [{ rotate: `${(1 - swing.value * 2) * SWING_DEG}deg` }],
  }));
  const turn = useAnimatedStyle(() => ({ transform: [{ rotate: `${rays.value * 360}deg` }] }));
  return { still, sun, moon, turn };
}

/** The round emblem with live bells: the boot screen, downloads. `size` in pt (the splash shows 200). */
export function FoolEmblem({ size = 200, style }: { size?: number; style?: StyleProp<ViewStyle> }) {
  const { still, sun, moon, turn } = useBells();
  const breath = useLoop(still, BREATH_MS, true);
  const breathe = useAnimatedStyle(() => ({ transform: [{ scale: 1 + 0.012 * breath.value }] }));

  const frame = (f: { left: number; top: number; size: number }) => ({
    left: f.left * size,
    top: f.top * size,
    width: f.size * size,
    height: f.size * size,
  });

  return (
    <Animated.View
      accessible
      accessibilityRole="progressbar"
      accessibilityLabel="Loading"
      style={[{ width: size, height: size }, breathe, style]}>
      <Image source={EMBLEM} style={styles.fill} />
      <Animated.View style={[styles.layer, frame(SUN_FRAME), { transformOrigin: SUN_PIVOT }, sun]}>
        <Animated.Image source={SUN_RAYS} style={[styles.fill, turn]} />
        <Image source={SUN_BODY} style={styles.fill} />
      </Animated.View>
      <Animated.View style={[styles.layer, frame(MOON_FRAME), { transformOrigin: MOON_PIVOT }, moon]}>
        <Image source={MOON} style={styles.fill} />
      </Animated.View>
    </Animated.View>
  );
}

/** Two small bells and a line of text: "The master is thinking…" in the chat. */
export function FoolBells({ label, size = 28 }: { label?: string; size?: number }) {
  const { sun, moon, turn } = useBells();
  // The moon's frame is wider than the sun's (180 vs 124 units); keep their bells the same size.
  const sunSide = size;
  const moonSide = size * (180 / 124);
  return (
    <View style={styles.inline} accessible accessibilityRole="progressbar" accessibilityLabel={label ?? 'Loading'}>
      <View style={styles.bells}>
        <Animated.View style={[{ width: sunSide, height: sunSide, transformOrigin: SUN_PIVOT }, sun]}>
          <Animated.Image source={SUN_RAYS} style={[styles.fill, turn]} />
          <Image source={SUN_BODY} style={styles.fill} />
        </Animated.View>
        <Animated.View style={[{ width: moonSide, height: moonSide, marginTop: -moonSide * 0.25, transformOrigin: MOON_PIVOT }, moon]}>
          <Image source={MOON} style={styles.fill} />
        </Animated.View>
      </View>
      {label ? (
        <ThemedText type="caption" themeColor="inkMuted" style={styles.label}>
          {label}
        </ThemedText>
      ) : null}
    </View>
  );
}

/** The emblem over a progress line and a caption: downloading a story (D.5–D.6). */
export function FoolProgress({ progress, label, size = 160 }: { progress?: number; label?: string; size?: number }) {
  const theme = useTheme();
  const share = progress === undefined ? undefined : Math.min(1, Math.max(0, progress));
  return (
    <View style={styles.progress}>
      <FoolEmblem size={size} />
      {share !== undefined ? (
        <View
          style={[styles.track, { width: size, backgroundColor: theme.line }]}
          accessibilityRole="progressbar"
          accessibilityValue={{ min: 0, max: 100, now: Math.round(share * 100) }}>
          <View style={[styles.bar, { width: share * size, backgroundColor: theme.accent }]} />
        </View>
      ) : null}
      {label ? (
        <ThemedText type="caption" themeColor="inkMuted">
          {label}
        </ThemedText>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { position: 'absolute', left: 0, top: 0, width: '100%', height: '100%' },
  layer: { position: 'absolute' },
  inline: { alignItems: 'center', flexDirection: 'row', gap: Spacing.two },
  bells: { alignItems: 'flex-start', flexDirection: 'row' },
  label: { fontStyle: 'italic' },
  progress: { alignItems: 'center', gap: Spacing.three },
  track: { borderRadius: Radius.pill, height: 3, overflow: 'hidden' },
  bar: { borderRadius: Radius.pill, height: 3 },
});
