/** On a phone the cards are swiped; an arrow only on the web (`.web.tsx`), where a mouse cannot swipe. */
export function CarouselArrow(_: { next: boolean; at: number; count: number; onGo: (index: number) => void }) {
  return null;
}
