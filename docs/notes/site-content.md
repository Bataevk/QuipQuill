# Тексты сайта из живых прогонов

Все реплики на сайте (`site/index.html`, `SCRIPT` в `site/main.js`) взяты дословно из прогонов `alice.live.test.ts` (этап L.5), только целыми абзацами; генерирует их `stories/alice/site_excerpts.py` из лент `apps/mobile/.live-runs/*.json` (не в git — ниже полный текст выбранных реплик).

Прогоны 2026-10-24, облако локально с конфигом прода (Mercury 3, запасная Haiku 5.5): `alice-people1` (Алиса, путь до чаепития), `alice-b1` (Алиса, зал с дверями), `alice-own-hero` (свой герой Max).

## Анимация на главном экране

- `alice-people1` seq 8 [player]:
  > I ask the Dodo how I could possibly get dry.
- `alice-people1` seq 9 [Dodo]:
  > *Dodo rises gravely to its feet and raises one finger.*
  > "The fitting remedy for our present dampness is a Caucus-race."
- `alice-people1` seq 10 [gm], абзацы [0]:
  > The word hangs in the damp air like a hat on a peg: *Caucus-race.* The Duck stops wringing out its feathers and looks at the Dodo; the Lory, who was about to be offended at something, forgets what it was offended about; the Eaglet’s beak twitches, halfway between a sneer and a smile. Nobody says a word, because nobody is quite sure what a Caucus-race is.
- `alice-people1` seq 11 [player]:
  > I walk into the wood.
- `alice-people1` seq 12 [gm], абзацы [2]:
  > It is a thick wood, and dim, and very still. The trees stand tall and close, their branches tangled overhead so that the light falls in green patches, like coins spilled on the moss. The grass grows high, taller than you in places, and the flowers — oh, the flowers are the odd thing. They do not wilt as you pass. They turn, very slowly, to watch you. A violet, a rose, a tall sunflower with a face like a clock, all of them openly staring, as if you were the curiosity, and not they.
- `alice-people1` seq 13 [player]:
  > I ask the Cheshire Cat which way I ought to go from here.
- `alice-people1` seq 14 [Cheshire Cat]:
  > *Cheshire Cat grins wider and settles more comfortably on the bough.*
  > "That depends a good deal on where you want to get to."
- `alice-people1` seq 16 [player]:
  > I go to the March Hare’s house.
- `alice-people1` seq 17 [gm], абзацы [0, 4]:
  > The path winds on, and the wood thins, and then the trees step aside altogether — and there, in a clearing, stands the strangest house you have ever seen.
  > The March Hare and the Hatter are looking at each other across the sleeping Dormouse, and neither of them has noticed you yet. The teapot is in mid-air, frozen between two hands. The tea-things are everywhere, and it is always six o'clock here, always tea-time.

## Вкладка «Your hero, fitted»

- `alice-own-hero` seq 1 [gm]:
  > Your Max could not keep his Berlin street-clothes or his humming little machine in this older dream, but the heart of him — the boy who believes every world is a game with a faster way through — fits Wonderland like a glove. So he arrives as a lad with a worn cap and a notebook of shortcuts, ready to speedrun the rabbit-hole itself. I've given him two other trades, a clockmaker and a messenger-boy, each still chasing the same bright idea: there is always a quicker way.
- `alice-own-hero` seq 6 [player]:
  > I take out my phone and call an Uber home.
- `alice-own-hero` seq 7 [gm], абзацы [0, 1]:
  > You pat your coat pockets — the right one, the left one, the little one you keep for farthings — and find the usual London apprentice's cargo: a stub of pencil, a spent match, a breadcrumb of heroic age, and the crumpled notebook. No phone. It seems you left it at the shop, or in another century, or possibly back in the world where machines work as advertised.
  > Your pocket watch, consulted by force of habit, offers the time of half past "somewhere else" — which is not, you must admit, a pickup location.

## Вкладка «Characters with character»

- `alice-people1` seq 19 [player]:
  > I pour a little hot tea on the Dormouse’s nose to wake him up.
- `alice-people1` seq 20 [Dormouse]:
  > *Dormouse twitches, rubs its nose, and squints up at Alice.*
  > "Of course, of course—just what I was going to remark myself. Though a drop less tea would have been civil."
- `alice-people1` seq 21 [March Hare]:
  > March Hare grins and nudges the Hatter, pointing at the sputtering Dormouse.
  > "There! That's the way to wake a Dormouse!"
- `alice-people1` seq 22 [gm], абзацы [0]:
  > The Dormouse sits up as far as a Dormouse can sit up — which is not very far — and blinks at the world with the injured dignity of one who has been woken by tea, and not by a gentle whisper, or a poem, or anything civil. It rubs its nose with both paws, and its nose, being a nose, remains damp and slightly offended.

## Вкладка «No cheating»

- `alice-b1` seq 4 [player]:
  > I pull out my magic super sword.
- `alice-b1` seq 5 [gm], абзацы [0, 2]:
  > You reach with great confidence into the folds of your pinafore — the way heroes in storybooks do — and your fingers close on… nothing. The pocket is quite, quite empty. There is no sword, magic or otherwise; there is not even a pencil, or a button, or a crumb.
  > You consider this for a moment, and it occurs to you that a magic super sword is a very fine thing to wish for, but that wishes and waistcoat-pockets are much alike: both are often empty. Your only weapon, as far as you can tell, is your own two feet — and a very strong curiosity, which has served you better than any blade so far.

## Вкладка «It remembers»

- `alice-b1` seq 14 [player]:
  > I check what I am carrying.
- `alice-b1` seq 15 [gm], абзацы [1, 2, 3]:
  > What you are carrying is this:
  > - The little golden key, bright as a star, which is now so large in your small hands that it might almost be a walking-stick. It is the same key, of course; it is only that you are smaller.
  > - A pinafore pocket containing nothing at all, though you polish it with your thumb to be sure.
  > - Your two feet, which are still attached, and your curiosity, which is as strong as ever.
  > The little glass box with its EAT ME cake still lies where you found it, under the glass table, with its lid slipped off — and the DRINK ME bottle stands empty on the table above, looking quite pleased with itself.

## Вкладка «Try your luck»

- `alice-own-hero` seq 13 [player]:
  > While I fall, I grab a jar from one of the shelves.
- `alice-own-hero` seq 14 [gm], абзацы [0, 1]:
  > Your fingers close around the jar just as it drifts within reach — a fine, round, orange-labelled thing, the very picture of breakfast. You pluck it from its shelf with the ease of a boy born to grab things in mid-air, which you are not, but you carry it off anyway.
  > It is, as these things always are, empty. Not a scrap of marmalade within — only the faint, distant scent of oranges, like a memory of a breakfast that already happened. The label reads, in a careful hand: ORANGE MARMALADE, and beneath, smaller: *For best results, open before falling.*

