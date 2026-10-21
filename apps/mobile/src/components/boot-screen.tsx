import * as SplashScreen from 'expo-splash-screen';
import { useEffect, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import Animated, { useAnimatedStyle, useSharedValue, withTiming } from 'react-native-reanimated';
import { scheduleOnRN } from 'react-native-worklets';

import { FoolHat } from '@/components/fool-loader';
import { useTheme } from '@/hooks/use-theme';

/** Must match `imageWidth` of the expo-splash-screen plugin (app.json): the handoff is seamless. */
const SPLASH_HAT = 200;
const FADE_MS = 300;

/**
 * The start (stage D.3): the native splash is held (`preventAutoHideAsync` in the root layout)
 * until this screen — the same hat's silhouette in the accent on the theme's ground, in the same
 * spot — is drawn; then the bells start to swing while the app opens its data, and the screen
 * fades away once `ready`.
 */
export function BootScreen({ ready }: { ready: boolean }) {
  const theme = useTheme();
  const [gone, setGone] = useState(false);
  const opacity = useSharedValue(1);

  useEffect(() => {
    if (!ready) return;
    opacity.value = withTiming(0, { duration: FADE_MS }, (finished) => {
      if (finished) scheduleOnRN(setGone, true);
    });
  }, [ready, opacity]);

  const fade = useAnimatedStyle(() => ({ opacity: opacity.value }));
  if (gone) return null;
  return (
    <Animated.View style={[StyleSheet.absoluteFill, fade]} pointerEvents={ready ? 'none' : 'auto'} onLayout={() => SplashScreen.hide()}>
      <View style={[styles.center, { backgroundColor: theme.bg }]}>
        <FoolHat width={SPLASH_HAT} />
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  center: { alignItems: 'center', flex: 1, justifyContent: 'center' },
});
