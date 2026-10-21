// foolore icon set (ADR 63): the old two-tone hat, mirrored onto a bottom-left → top-right diagonal,
// with sun and moon bells and a faint paper grain. Writes SVGs to icons/foolore/ and previews to
// icons/foolore/preview/; render.mjs turns the SVGs into app assets. usage (from icons/): npm run render
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { Resvg } from '@resvg/resvg-js';
import sharp from 'sharp';

const OUT = 'foolore';
const PREVIEW = join(OUT, 'preview');
mkdirSync(PREVIEW, { recursive: true });

// Brighter than the app's accent (#A3313A): a home-screen icon sits next to saturated ones, and
// the first set (#B8323B, #2E2A30, #D9A441, a heavier grain) looked dull there (2026-10-21).
const C = {
  red: '#D9343F',
  black: '#332E36',
  gold: '#F2BD4F',
  goldDark: '#9C6F22',
  cream: '#EFE3C8',
};

// The owner's hat (icons/QuitQuill-square.svg), in its original group coordinates.
const LOBE = [
  'M -279.4 9.656 C -278.3 7.114 -259.51 -29.718 -256.34 -34.616 C -218.98 -92.518 -157.52 -131.45 -86.72 -115.73 C -31.59 -103.48 22.666 -48.34 27.544 9.628 L 26.64 21.593 C 26.172 21.401 12.832 -14.144 10.174 -17.672 C -13.086 -48.544 -53.318 -61.3 -89.288 -46.038 C -151.12 -19.806 -150.79 49.036 -163.68 103.81 C -166.3 114.93 -170.63 124.854 -174.48 135.506 C -209.32 98.562 -278.98 11.51 -279.4 9.656 Z',
  'M -175.48 132.506 C -171.63 121.854 -167.3 111.93 -164.68 100.81 C -151.79 46.036 -152.12 -22.806 -90.288 -49.038 C -54.318 -64.3 -14.086 -51.544 9.174 -20.672 C 11.832 -17.144 25.612 8.058 26.08 8.25 L 26.544 6.628 C 27.158 12.588 27.044 18.056 26.544 24.006 C -9.156 -53.67 -98.34 -55.47 -127.45 28.05 C -142.35 70.786 -137.54 115.392 -125.29 157.932 C -124.61 160.308 -121.17 166.492 -123.67 168.19 C -127.37 167.24 -130.95 165.922 -134.69 165.042 C -138.57 161.52 -144.3 159.768 -148.68 156.836 C -158.33 150.386 -167.84 141.238 -175.48 132.506 Z',
];
const BODY = 'M -44.13 216.069 C -38.92 142.887 22.57 84.817 93.65 74.225 C 142.38 66.963 178.139 77.045 222.55 109.267 C 291.027 159.349 304.391 193.563 336.26 229.737 C 343.9 238.469 353.41 247.617 363.06 254.067 C 367.44 256.999 373.17 258.751 377.05 262.273 C 358.52 262.221 342.23 262.733 323.84 265.421 C 286.37 272.203 249.76 280.787 214 294.101 C 192.68 302.131 169.71 312.047 151.97 326.565 C 152.18 275.923 116.19 212.925 73.95 186.893 C 31.15 160.517 -11.32 174.117 -36.57 216.069 C -39.09 215.775 -41.6 215.965 -44.13 216.069 Z';
const BRIM = 'M 323.84 265.421 C 342.23 262.733 380.72 250.833 411.431 263.844 C 430.993 273.937 439.406 313.999 413.646 338.549 C 409.26 342.725 399.8 350.249 393.54 350.225 C 392.36 349.889 261.729 384.89 187.436 401.869 C 157.849 400.483 124 356.449 151.97 326.565 C 169.71 312.047 192.68 302.131 214 294.101 C 249.76 280.787 286.37 272.203 323.84 265.421 Z';
const OPENING = 'M 180.91 397.419 C 184.526 375.377 259.36 354.451 273.2 350.585 C 304.46 341.853 347.522 332.883 380.25 333.394 C 391.087 333.563 400.49 339.681 393.54 350.225 C 337.602 390.392 176.409 419.946 180.91 397.419 Z';

// Bell positions after mirroring (1024 canvas): sun on the red (top-left) side, moon on the black.
// Each bell hangs from its tip: the centre sits one radius below it.
const SUN_TIP = { x: 222, y: 419 };
const MOON_TIP = { x: 801, y: 518 };

const stroke = (on, color, width) => (on ? ` stroke="${color}" stroke-width="${width}" stroke-linejoin="round"` : '');

// A part of the hat: its outline is a doubled stroke drawn under the fills, so only the outer edge shows.
const part = (paths, fill, line, width, outline) =>
  (outline ? paths.map((d) => `<path d="${d}" fill="${line}" stroke="${line}" stroke-width="${width * 2}" stroke-linejoin="round"/>`).join('') : '') +
  paths.map((d) => `<path d="${d}" fill="${fill}"/>`).join('');

function hat({ outline }) {
  // Mirrored around the canvas's vertical axis; colours are the reverse of the ground behind them.
  return `<g transform="translate(1024 0) scale(-1 1)">
  <g transform="translate(265.997349 297.000004)">
    <g transform="translate(511.740181 97.230699)">${part(LOBE, C.black, C.red, 4, outline)}</g>
    ${part([BODY], C.red, C.black, 4, outline)}
    ${part([BRIM], C.gold, C.goldDark, 3.5, outline)}
    ${part([OPENING], C.black, C.red, 3, outline)}
  </g>
</g>`;
}

function sun({ outline, face, bell = 1, part = 'all' }) {
  const r = 33 * bell, SUN = { x: SUN_TIP.x, y: SUN_TIP.y + r };
  const rays = Array.from({ length: 8 }, (_, i) => {
    const a = (i * Math.PI) / 4;
    const p = (rad, da) => `${(SUN.x + rad * Math.sin(a + da)).toFixed(1)} ${(SUN.y - rad * Math.cos(a + da)).toFixed(1)}`;
    return `M ${p(r * 1.75, 0)} L ${p(r * 1.05, 0.2)} L ${p(r * 1.05, -0.2)} Z`;
  }).join(' ');
  const details = face
    ? `<circle cx="${SUN.x - r * 0.32}" cy="${SUN.y - r * 0.15}" r="${r * 0.1}" fill="${C.goldDark}"/>
  <circle cx="${SUN.x + r * 0.32}" cy="${SUN.y - r * 0.15}" r="${r * 0.1}" fill="${C.goldDark}"/>
  <path d="M ${SUN.x - r * 0.38} ${SUN.y + r * 0.22} Q ${SUN.x} ${SUN.y + r * 0.58} ${SUN.x + r * 0.38} ${SUN.y + r * 0.22}" fill="none" stroke="${C.goldDark}" stroke-width="${r * 0.1}" stroke-linecap="round"/>`
    : '';
  // The loader turns the rays on their own (stage D.3): `part` draws them or the face alone.
  const rayPath = `<path fill="${C.gold}" d="${rays}"${stroke(outline, C.goldDark, 4)}/>`;
  const body = `<circle cx="${SUN.x}" cy="${SUN.y}" r="${r}" fill="${C.gold}"${stroke(outline, C.goldDark, 5)}/>
  ${details}`;
  return `<g>
  ${part === 'body' ? '' : rayPath}
  ${part === 'rays' ? '' : body}
</g>`;
}

function moon({ outline, face, bell = 1 }) {
  // Crescent opening up and to the right: outer circle minus a circle shifted up-right.
  const r = 42 * bell, cx = MOON_TIP.x, cy = MOON_TIP.y + r, ix = cx + r * 0.5, iy = cy - r * 0.36, ir = r * 0.74;
  const star = (x, y, s) => `M ${x} ${y - s} L ${x + s * 0.28} ${y - s * 0.28} L ${x + s} ${y} L ${x + s * 0.28} ${y + s * 0.28} L ${x} ${y + s} L ${x - s * 0.28} ${y + s * 0.28} L ${x - s} ${y} L ${x - s * 0.28} ${y - s * 0.28} Z`;
  const details = face
    ? `<path d="M ${cx - r * 0.55} ${cy + r * 0.05} q ${r * 0.12} ${r * 0.1} ${r * 0.24} 0" fill="none" stroke="${C.goldDark}" stroke-width="${r * 0.08}" stroke-linecap="round"/>
  <path d="M ${cx - r * 0.42} ${cy + r * 0.42} q ${r * 0.2} ${r * 0.12} ${r * 0.36} -${r * 0.06}" fill="none" stroke="${C.goldDark}" stroke-width="${r * 0.08}" stroke-linecap="round"/>`
    : '';
  return `<g>
  <mask id="crescent" maskUnits="userSpaceOnUse" x="0" y="0" width="1024" height="1024">
    <rect width="1024" height="1024" fill="#fff"/>
    <circle cx="${ix}" cy="${iy}" r="${ir}" fill="#000"/>
  </mask>
  <circle cx="${cx}" cy="${cy}" r="${r}" fill="${C.gold}" mask="url(#crescent)"/>
  ${outline ? `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${C.goldDark}" stroke-width="5" mask="url(#crescent)"/>
  <circle cx="${ix}" cy="${iy}" r="${ir}" fill="none" stroke="${C.goldDark}" stroke-width="5" clip-path="url(#moonClip)"/>
  <clipPath id="moonClip"><circle cx="${cx}" cy="${cy}" r="${r}"/></clipPath>` : ''}
  <path fill="${C.gold}" d="${star(cx + r * 0.9, cy - r * 0.05, r * 0.36)}"${stroke(outline, C.goldDark, 3)}/>
  ${details}
</g>`;
}

const GRAIN = `<filter id="grain" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse" x="0" y="0" width="1024" height="1024">
  <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" seed="5" stitchTiles="stitch" result="fine"/>
  <feColorMatrix in="fine" type="saturate" values="0" result="fineMono"/>
  <feComponentTransfer in="fineMono" result="fineSoft">
    <feFuncR type="linear" slope="0.1" intercept="0.94"/><feFuncG type="linear" slope="0.1" intercept="0.94"/>
    <feFuncB type="linear" slope="0.1" intercept="0.94"/><feFuncA type="linear" slope="0" intercept="1"/>
  </feComponentTransfer>
  <feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves="3" seed="11" result="mottle"/>
  <feColorMatrix in="mottle" type="saturate" values="0" result="mottleMono"/>
  <feComponentTransfer in="mottleMono" result="mottleSoft">
    <feFuncR type="linear" slope="0.04" intercept="0.97"/><feFuncG type="linear" slope="0.04" intercept="0.97"/>
    <feFuncB type="linear" slope="0.04" intercept="0.97"/><feFuncA type="linear" slope="0" intercept="1"/>
  </feComponentTransfer>
  <feBlend in="SourceGraphic" in2="fineSoft" mode="multiply" result="a"/>
  <feBlend in="a" in2="mottleSoft" mode="multiply" result="b"/>
  <feComposite in="b" in2="SourceGraphic" operator="in"/>
</filter>`;

const ground = () => `<g filter="url(#grain)">
  <rect width="1024" height="1024" fill="${C.black}"/>
  <path d="M 0 0 L 1024 0 L 0 1024 Z" fill="${C.red}"/>
</g>`;

// Android themed icon: one tint, so parts are told apart by gaps. Mask = parts in white, their
// outline underlays in black; colours inside the crescent's own <mask> stay as they are.
function monochrome(body) {
  const parts = body.split(/(<mask[\s\S]*?<\/mask>)/);
  const tint = (chunk) =>
    chunk.replace(/<(path|circle)\b[^>]*>/g, (tag) => {
      const fill = tag.match(/fill="([^"]+)"/)?.[1];
      const line = tag.match(/stroke="([^"]+)"/)?.[1];
      let out = tag.replace(/stroke="[^"]+"/, 'stroke="#000"');
      if (fill && fill !== 'none') out = out.replace(/fill="[^"]+"/, `fill="${line && fill === line ? '#000' : '#fff'}"`);
      return out;
    });
  const masked = parts.map((chunk) => (chunk.startsWith('<mask') ? chunk : tint(chunk))).join('');
  return `<mask id="mono" maskUnits="userSpaceOnUse" x="0" y="0" width="1024" height="1024">${masked}</mask>
<rect width="1024" height="1024" fill="#000" mask="url(#mono)"/>`;
}

// One colour for every part (the boot silhouette): fills outside the crescent's <mask> become `color`.
function silhouette(body, color) {
  return body
    .split(/(<mask[\s\S]*?<\/mask>)/)
    // A hairline stroke of the same colour closes the antialiased seams where parts meet.
    .map((chunk) =>
      chunk.startsWith('<mask')
        ? chunk
        : chunk.replace(/fill="(?!none)[^"]+"/g, `fill="${color}" stroke="${color}" stroke-width="2" stroke-linejoin="round"`),
    )
    .join('');
}

// Paper for backgrounds (ADR 63) in two layers over the theme's bg: a small tile of fine grain,
// repeated, and soft mottling as one small image stretched over the screen (low-frequency, so it
// stays smooth when stretched and leaves no seams to show). Both are alpha-only: dark specks on
// parchment, light ones at night; together they shift the mean to the splash's colour (D.2).
function speckle({ width, height, frequency, octaves, seed, light, k, b }) {
  const rgb = light ? '0' : '1';
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">
<defs><filter id="s" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse" x="0" y="0" width="${width}" height="${height}">
  <feTurbulence type="fractalNoise" baseFrequency="${frequency}" numOctaves="${octaves}" seed="${seed}" stitchTiles="stitch"/>
  <feColorMatrix type="matrix" values="0 0 0 0 ${rgb}  0 0 0 0 ${rgb}  0 0 0 0 ${rgb}  ${k} 0 0 0 ${b}"/>
</filter></defs>
<rect width="${width}" height="${height}" fill="#000" filter="url(#s)"/>
</svg>
`;
}
const grain = (light) =>
  speckle({ width: 256, height: 256, frequency: 0.9, octaves: 2, seed: 5, light, ...(light ? { k: 0.07, b: -0.012 } : { k: 0.035, b: -0.005 }) });
const mottle = (light) =>
  speckle({ width: 160, height: 346, frequency: 0.03, octaves: 3, seed: 11, light, ...(light ? { k: 0.06, b: -0.016 } : { k: 0.03, b: -0.007 }) });

const frame = `<rect x="54" y="54" width="916" height="916" rx="176" fill="none" stroke="${C.cream}" stroke-width="10" opacity="0.85"/>`;

const svg = (body, viewBox = '0 0 1024 1024', defs = '') =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}"><defs>${GRAIN}${defs}</defs>${body}</svg>\n`;

const art = (o) => `${hat(o)}\n${sun(o)}\n${moon(o)}`;

// Scale the drawing about the centre (the hat fills ~60% of the canvas; icons want a little air).
const scaled = (body, s) => `<g transform="translate(512 520) scale(${s}) translate(-512 -512)">${body}</g>`;

// The app's accent (src/constants/theme.ts) and the silhouette's frame: the art's own units.
const ACCENT = { light: '#A3313A', dark: '#E06C70' };
// The hat with its bells spans x 165–853, y 275–703; a little air around it.
const SILHOUETTE_BOX = '150 255 724 470';

const files = {
  'icon-a.svg': svg(`${ground()}${scaled(art({ outline: false, face: false, bell: 1.3 }), 1.0)}`),
  'icon-b.svg': svg(`${ground()}${frame}${scaled(art({ outline: false, face: false, bell: 1.3 }), 0.92)}`),
  // Large emblem: round badge with outlines and faces (splash, loading, onboarding).
  'emblem.svg': svg(
    `<g clip-path="url(#disc)">${ground()}</g>
<circle cx="512" cy="512" r="440" fill="none" stroke="${C.cream}" stroke-width="10" opacity="0.9"/>
${scaled(art({ outline: true, face: true }), 0.95)}`,
    '62 62 900 900',
    '<clipPath id="disc"><circle cx="512" cy="512" r="445"/></clipPath>',
  ),
  // The loader (stage D.3): the emblem without its bells, and the bells on their own, each framed
  // so the app can swing it about the hat's tip. Frames in the art's units (before the 0.95):
  // sun — 124×124 about its centre (rays turn there), moon — 180×180 about its tip.
  'emblem-base.svg': svg(
    `<g clip-path="url(#disc)">${ground()}</g>
<circle cx="512" cy="512" r="440" fill="none" stroke="${C.cream}" stroke-width="10" opacity="0.9"/>
${scaled(hat({ outline: true }), 0.95)}`,
    '62 62 900 900',
    '<clipPath id="disc"><circle cx="512" cy="512" r="445"/></clipPath>',
  ),
  'bell-sun-rays.svg': svg(sun({ outline: true, face: true, part: 'rays' }), '160 390 124 124'),
  'bell-sun-body.svg': svg(sun({ outline: true, face: true, part: 'body' }), '160 390 124 124'),
  'bell-moon.svg': svg(moon({ outline: true, face: true }), '711 428 180 180'),
  // App layers (variant A, owner's choice 2026-10-20): ground and hat apart for Android.
  'ground.svg': svg(ground()),
  'hat.svg': svg(scaled(art({ outline: false, face: false, bell: 1.3 }), 1.0)),
  'hat-mono.svg': svg(monochrome(scaled(art({ outline: true, face: false, bell: 1.3 }), 1.0))),
  'paper-grain-light.svg': grain(true),
  'paper-grain-dark.svg': grain(false),
  'paper-mottle-light.svg': mottle(true),
  'paper-mottle-dark.svg': mottle(false),
  // Hat and bells alone, outlined, for the loader and screens with their own background.
  'hat-outlined.svg': svg(art({ outline: true, face: true }), '150 180 724 520'),
  // The start (2026-10-21): the hat's silhouette in the theme's accent on the theme's ground —
  // for the native splash in both themes, and in white for the app to tint (the boot screen's
  // layers: the hat without bells, and each bell in the frame the loader swings it in).
  'splash-light.svg': svg(silhouette(art({ outline: false, face: false }), ACCENT.light), SILHOUETTE_BOX),
  'splash-dark.svg': svg(silhouette(art({ outline: false, face: false }), ACCENT.dark), SILHOUETTE_BOX),
  'silhouette-hat.svg': svg(silhouette(hat({ outline: false }), '#fff'), SILHOUETTE_BOX),
  'silhouette-sun-rays.svg': svg(silhouette(sun({ outline: false, face: false, part: 'rays' }), '#fff'), '160 390 124 124'),
  'silhouette-sun-body.svg': svg(silhouette(sun({ outline: false, face: false, part: 'body' }), '#fff'), '160 390 124 124'),
  'silhouette-moon.svg': svg(silhouette(moon({ outline: false, face: false }), '#fff'), '711 428 180 180'),
};
for (const [name, text] of Object.entries(files)) writeFileSync(join(OUT, name), text);

async function png(svgText, file, size) {
  const raw = new Resvg(svgText, { fitTo: { mode: 'width', value: size } }).render().asPng();
  writeFileSync(join(PREVIEW, file), await sharp(raw).png().toBuffer());
}

// Previews: each icon at home-screen sizes, plus a sheet with iOS-style rounded masks.
for (const v of ['a', 'b']) for (const s of [1024, 180, 60, 48]) await png(files[`icon-${v}.svg`], `icon-${v}-${s}.png`, s);
await png(files['emblem.svg'], 'emblem-600.png', 600);
await png(files['hat-outlined.svg'], 'hat-outlined-600.png', 600);
await png(files['hat-mono.svg'], 'hat-mono-600.png', 600);

const tile = async (file, size) => {
  const img = await sharp(join(PREVIEW, file)).resize(size, size).png().toBuffer();
  const mask = Buffer.from(`<svg width="${size}" height="${size}"><rect width="${size}" height="${size}" rx="${size * 0.225}" fill="#fff"/></svg>`);
  return sharp(img).composite([{ input: mask, blend: 'dest-in' }]).png().toBuffer();
};
for (const [bg, name] of [['#F4ECDD', 'sheet-light.png'], ['#1D1A1E', 'sheet-dark.png']]) {
  const parts = [];
  let x = 24;
  for (const v of ['a', 'b']) {
    for (const s of [180, 120, 60, 48]) {
      parts.push({ input: await tile(`icon-${v}-1024.png`, s), left: x, top: 24 + (v === 'b' ? 220 : 0) + (180 - s) / 2 });
      x += s + 24;
    }
    x = 24;
  }
  await sharp({ create: { width: 24 * 5 + 180 + 120 + 60 + 48, height: 24 * 3 + 180 * 2 + 40, channels: 3, background: bg } })
    .composite(parts)
    .png()
    .toFile(join(PREVIEW, name));
}
console.log('done');
