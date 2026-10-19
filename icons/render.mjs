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

// Paper tiles for backgrounds outside the main pages (ADR 63), repeated at 1x.
await png(read('paper-light.svg'), 'paper-light.png', 512);
await png(read('paper-dark.svg'), 'paper-dark.png', 512);
