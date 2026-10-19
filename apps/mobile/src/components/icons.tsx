/** Line icons for the tab bar (stage D.7): one stroke weight, round joins, drawn on a 24 grid. */
import type { ColorValue } from 'react-native';
import Svg, { Circle, Path } from 'react-native-svg';

type IconProps = { color: ColorValue; size?: number };

const line = (color: ColorValue) => ({
  stroke: color,
  strokeWidth: 1.8,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  fill: 'none',
});

/** An open book: the stories. */
export function BookIcon({ color, size = 24 }: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      <Path d="M12 6.5C10 5 7.2 4.5 3.5 4.8v13.4c3.7-.3 6.5.2 8.5 1.8 2-1.6 4.8-2.1 8.5-1.8V4.8C16.8 4.5 14 5 12 6.5Z" {...line(color)} />
      <Path d="M12 6.5V20" {...line(color)} />
    </Svg>
  );
}

/** A player's mask: the heroes you play. */
export function MaskIcon({ color, size = 24 }: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      <Path d="M4 6.5c5.3-1.8 10.7-1.8 16 0 0 7-2.7 12-8 12.5-5.3-.5-8-5.5-8-12.5Z" {...line(color)} />
      <Path d="M7.4 10.2c.9-.7 2.1-.7 3 0M13.6 10.2c.9-.7 2.1-.7 3 0" {...line(color)} />
      <Path d="M9.2 14.6c1.8 1.2 3.8 1.2 5.6 0" {...line(color)} />
    </Svg>
  );
}

/** A person in a circle: the account and settings. */
export function ProfileIcon({ color, size = 24 }: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      <Circle cx={12} cy={12} r={8.5} {...line(color)} />
      <Circle cx={12} cy={10} r={2.8} {...line(color)} />
      <Path d="M6.6 17.6c1.3-2 3.1-3 5.4-3s4.1 1 5.4 3" {...line(color)} />
    </Svg>
  );
}
