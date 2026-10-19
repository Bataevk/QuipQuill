// Renders the app's icons, splash and paper tiles from the foolore SVGs (foolore.mjs, ADR 63).
// usage (from icons/): npm install && npm run render
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { Resvg } from '@resvg/resvg-js';
import sharp from 'sharp';

const [src, out] = process.argv.slice(2);
const read = (f) => readFileSync(join(src, f), 'utf8');

async function png(svgText, file, size, { flatten } = {}) {
  const raw = new Resvg(svgText, { fitTo: { mode: 'width', value: size } }).render().asPng();
  let img = sharp(raw);
  if (flatten) img = img.flatten({ background: flatten }).removeAlpha();
  writeFileSync(join(out, file), await img.png().toBuffer());
  console.log(file, size);
}

// Scales an icon SVG's drawing about the centre (Android shows only the inner ~2/3 of its canvas).
const shrink = (svgText, s) =>
  svgText.replace(/(<\/defs>)([\s\S]*)(<\/svg>)/, `$1<g transform="translate(512 512) scale(${s}) translate(-512 -512)">$2</g>$3`);

// iOS / generic icon: full square, no alpha (App Store rejects alpha).
await png(read('icon-a.svg'), 'icon.png', 1024, { flatten: '#2E2A30' });

// Android adaptive (108dp canvas, ~72dp visible): hat at 2/3 so it looks like the iOS icon.
const ANDROID_HAT = 0.667;
await png(read('ground.svg'), 'android-icon-background.png', 1024);
await png(shrink(read('hat.svg'), ANDROID_HAT), 'android-icon-foreground.png', 1024);
await png(shrink(read('hat-mono.svg'), ANDROID_HAT), 'android-icon-monochrome.png', 1024);

// Splash: the outlined round emblem (Android 12+ shows it inside a circle).
await png(read('emblem.svg'), 'splash-icon.png', 1024);

// Web favicon: the icon itself reads better than the emblem at 48 px.
await png(read('icon-a.svg'), 'favicon.png', 48);

// The loader's layers (stage D.3): the emblem without bells at 3x of 200 pt, the bells larger
// than they show at 200 pt so the same files serve bigger emblems too.
await png(read('emblem-base.svg'), 'loader-emblem.png', 768);
await png(read('bell-sun-rays.svg'), 'loader-sun-rays.png', 192);
await png(read('bell-sun-body.svg'), 'loader-sun-body.png', 192);
await png(read('bell-moon.svg'), 'loader-moon.png', 256);

// Paper for backgrounds outside the main pages (ADR 63): a grain tile and a stretched mottle.
for (const t of ['light', 'dark']) {
  await png(read(`paper-grain-${t}.svg`), `paper-grain-${t}.png`, 256);
  await png(read(`paper-mottle-${t}.svg`), `paper-mottle-${t}.png`, 160);
}
