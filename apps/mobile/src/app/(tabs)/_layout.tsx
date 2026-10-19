import { Tabs } from 'expo-router';

import { BookIcon, MaskIcon, ProfileIcon } from '@/components/icons';
import { Fonts } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

/** The main pages (stage D.7): stories first, then heroes, then the profile. Flat ground (ADR 63). */
export default function TabsLayout() {
  const theme = useTheme();
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: theme.accent,
        tabBarInactiveTintColor: theme.inkMuted,
        tabBarStyle: { backgroundColor: theme.surface, borderTopColor: theme.line },
        tabBarLabelStyle: { fontFamily: Fonts.sans, fontSize: 11, fontWeight: 600 },
        headerStyle: { backgroundColor: theme.bg },
        headerShadowVisible: false,
        headerTitleStyle: { fontFamily: Fonts.display, fontSize: 22, color: theme.ink },
      }}>
      <Tabs.Screen
        name="index"
        options={{ title: 'Stories', headerTitle: 'foolore', tabBarIcon: ({ color }) => <BookIcon color={color} /> }}
      />
      <Tabs.Screen name="heroes" options={{ title: 'Heroes', tabBarIcon: ({ color }) => <MaskIcon color={color} /> }} />
      <Tabs.Screen name="profile" options={{ title: 'Profile', tabBarIcon: ({ color }) => <ProfileIcon color={color} /> }} />
    </Tabs>
  );
}
