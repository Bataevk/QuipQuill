// Renders app icons and splash from the owner's SVGs in <repo>/icons.
// usage (from icons/): npm install && npm run render
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { Resvg } from '@resvg/resvg-js';
import sharp from 'sharp';

const [src, out] = process.argv.slice(2);
const read = (f) => readFileSync(join(src, f), 'utf8');

// Inner <g> of an SVG (the 1024 viewBox drawing) and its paths.
const inner = (svg) => svg.slice(svg.indexOf('<g'), svg.lastIndexOf('</svg>'));
const square = inner(read('QuitQuill-square.svg'));
const hat = inner(read('logo.svg'));
const round = inner(read('QuitQuill.svg'));

// Background of the square icon = its first two paths (diagonal red / dark), no hat.
const outerOpen = square.slice(0, square.indexOf('>') + 1);
const bgPaths = square.match(/<path[^>]*\/>/g).slice(0, 2).join('\n');
const background = `${outerOpen}\n${bgPaths}\n</g>`;

const svg = (body, { viewBox = '0 0 1024 1024', bg = '' } = {}) =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}">${bg}${body}</svg>`;
const scaled = (body, s) =>
  `<g transform="translate(512 512) scale(${s}) translate(-512 -512)">${body}</g>`;

async function png(svgText, file, size, { flatten } = {}) {
  const raw = new Resvg(svgText, { fitTo: { mode: 'width', value: size } }).render().asPng();
  let img = sharp(raw);
  if (flatten) img = img.flatten({ background: flatten }).removeAlpha();
  writeFileSync(join(out, file), await img.png().toBuffer());
  console.log(file, size);
}

// iOS / generic icon: full square, no alpha (App Store rejects alpha).
await png(svg(square), 'icon.png', 1024, { flatten: '#3C3A40' });

// Android adaptive (108dp canvas, ~72dp visible): hat at 2/3 so it looks like the iOS icon.
const ANDROID_HAT = 0.667;
await png(svg(background), 'android-icon-background.png', 1024);
await png(svg(scaled(hat, ANDROID_HAT)), 'android-icon-foreground.png', 1024);
const silhouette = hat
  .replace(/fill="#[0-9A-Fa-f]{6}"/g, 'fill="#000000"')
  .replace(/fill: rgb\([^)]*\)/g, 'fill: #000000')
  .replace(/<path d=/g, '<path fill="#000000" d=');
await png(svg(scaled(silhouette, ANDROID_HAT)), 'android-icon-monochrome.png', 1024);

// Splash: the round badge, cropped to its circle (approx. bbox of QuitQuill-512.png).
await png(svg(round, { viewBox: '112 112 800 800' }), 'splash-icon.png', 1024);

// Web favicon.
await png(svg(round, { viewBox: '112 112 800 800' }), 'favicon.png', 48);
