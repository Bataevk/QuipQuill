"""Writes stories/alice/curation.toml: hand-picked merges, renames and map; the drop list is
everything outside the kept places, cast and props (generated, so it stays exhaustive)."""
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
graph = json.loads((root / "graph.json").read_text())
names = {e["name"] for e in graph["entities"]}

MERGE = {
    # places
    "bank": ["field", "hedge"],
    "rabbit-hole": ["deep well", "The Well", "Centre of the Earth"],
    "Long Hall": ["dark hall", "the hall", "great hall", "Long Passage"],
    "pool of tears": ["large pool of tears", "The Pool"],
    "shore": ["race-course"],
    "neat little house": ["house"],
    "wood": ["the thick wood", "trees"],
    "little house": ["open place"],
    "The Queen's Croquet-Ground": ["Croquet-ground", "garden", "lovely garden", "beautiful garden"],
    "sea-shore": ["ledge of rock", "shingle"],
    "court of justice": ["court"],
    # people
    "White Rabbit": ["Rabbit"],
    "Bill the Lizard": ["Bill", "Lizard"],
    "Cheshire cat": ["Cat's head", "grin"],
    "Queen of Hearts": ["Queen"],
    "King of Hearts": ["King"],
    "Knave of Hearts": ["Knave"],
    "Frog-Footman": ["Footman", "other footman"],
    "Baby": ["pig-baby"],
    "Cook": ["Duchess's cook"],
    "Alice's sister": ["sister"],
    # things
    "little golden key": ["tiny golden key", "key"],
    "little bottle": ["bottle", "little magic bottle"],
    "white kid gloves": ["Rabbit's little white kid gloves", "pair of white kid gloves", "pair of gloves", "gloves"],
    "glass table": ["three-legged table", "Little glass table"],
    "mushroom": ["right-hand bit", "lefthand bit", "pieces of mushroom", "lefthand bit of mushroom"],
    "comfits": ["box of comfits"],
    "pepper-box": ["pepper"],
    "cauldron of soup": ["Soup"],
    "tea-things": ["tea", "clean cup", "milk-jug", "teacup", "tea spoon"],
    "bread-and-butter": ["piece of bread-and-butter", "butter"],
    "Live flamingoes": ["flamingo"],
    "Live hedgehogs": ["hedgehog", "other hedgehog"],
    "King's crown": ["crown"],
    "dish of tarts": ["tarts"],
    "scroll of parchment": ["parchment scroll"],
}

KEEP = set(MERGE) | {
    "Tidy Little Room", "large mushroom", "Kitchen", "house of the March Hare", "Tree with door",
    "Alice's sister", "Mouse", "Dodo", "Duck", "Lory", "Eaglet", "Caterpillar", "Pigeon", "Duchess",
    "Fish-Footman", "Hatter", "March Hare", "Dormouse", "Two", "Five", "Seven", "Gryphon",
    "Mock Turtle", "executioner",
    "watch", "daisies", "book", "ORANGE MARMALADE jar", "cupboards", "book-shelves",
    "little glass box", "cake", "fan", "thimble", "hookah", "Frying-pan", "table under tree",
    "large arm-chair", "White Roses", "trumpet", "hat", "sixpence",
}
KEEP.discard("White Roses")
KEEP.add("White roses")

RENAME = {
    "bank": "Riverbank",
    "rabbit-hole": "The Rabbit-Hole",
    "Long Hall": "Hall of Doors",
    "pool of tears": "Pool of Tears",
    "shore": "Shore of the Pool",
    "neat little house": "White Rabbit's House",
    "wood": "The Wood",
    "large mushroom": "Caterpillar's Mushroom",
    "little house": "Duchess's House",
    "Kitchen": "Duchess's Kitchen",
    "house of the March Hare": "March Hare's House",
    "Tree with door": "Tree with a Door",
    "The Queen's Croquet-Ground": "The Queen's Garden",
    "sea-shore": "Sea-Shore",
    "court of justice": "Court of Justice",
    "Cheshire cat": "Cheshire Cat",
    "cupboards": "cupboards and book-shelves",
    "little bottle": "DRINK ME bottle",
    "cake": "EAT ME cake",
}
MERGE_EXTRA = {"cupboards": ["book-shelves"]}

PASSAGES = [
    ("Riverbank", "The Rabbit-Hole", "A large rabbit-hole under the hedge; it dips suddenly down into a very deep well."),
    ("The Rabbit-Hole", "Hall of Doors", "The long fall ends on a heap of sticks and dry leaves, and a passage leads to a long, low hall."),
    ("Hall of Doors", "Pool of Tears", "Past the far end of the hall the floor slopes down into a wide salt pool: the Pool of Tears, deep enough to swim in."),
    ("Pool of Tears", "Shore of the Pool", "The pool's bank, crowded with dripping birds and animals."),
    ("Shore of the Pool", "The Wood", "Past the bank, a thick wood begins."),
    ("The Wood", "White Rabbit's House", "A neat little house with a brass plate: W. RABBIT."),
    ("The Wood", "Caterpillar's Mushroom", "Among the flowers and grass stands a mushroom as tall as a little girl."),
    ("The Wood", "Duchess's House", "An open place with a little house about four feet high."),
    ("The Wood", "March Hare's House", "A house with chimneys shaped like ears and a roof thatched with fur."),
    ("The Wood", "Tree with a Door", "One of the trees has a door leading right into it."),
    ("Tree with a Door", "Hall of Doors", "Through the door in the tree: the long hall again, close to the little glass table."),
    ("Hall of Doors", "The Queen's Garden", "The little door, fifteen inches high, opens with the golden key — for someone small enough."),
    ("The Queen's Garden", "Sea-Shore", "The Gryphon knows the way to the Mock Turtle's ledge by the sea."),
    ("The Queen's Garden", "Court of Justice", "The trial is beginning: everyone hurries to the court."),
]
INSIDE = [
    ("Tidy Little Room", "White Rabbit's House", "Upstairs in the Rabbit's house."),
    ("Duchess's Kitchen", "Duchess's House", "The door leads right into a large kitchen, full of smoke."),
]

# Where everyone and everything starts (names after renames). The extractor placed many of them
# in several places at once (the whole book); those ties are dropped and replaced by these.
PLACE = {
    "Riverbank": ["Alice's sister", "White Rabbit", "daisies", "book"],
    "The Rabbit-Hole": ["cupboards and book-shelves"],
    "Hall of Doors": ["glass table", "little golden key", "DRINK ME bottle", "little glass box"],
    "Pool of Tears": ["Mouse"],
    "Shore of the Pool": ["Dodo", "Duck", "Lory", "Eaglet", "comfits", "thimble"],
    "White Rabbit's House": ["Bill the Lizard"],
    "Tidy Little Room": ["fan", "white kid gloves"],
    "The Wood": ["Pigeon", "Cheshire Cat"],
    "Caterpillar's Mushroom": ["Caterpillar", "mushroom"],
    "Duchess's House": ["Frog-Footman", "Fish-Footman"],
    "Duchess's Kitchen": ["Duchess", "Cook", "Baby", "pepper-box", "cauldron of soup", "Frying-pan"],
    "March Hare's House": ["Hatter", "March Hare", "Dormouse", "table under tree", "tea-things", "bread-and-butter", "large arm-chair"],
    "The Queen's Garden": ["Queen of Hearts", "King of Hearts", "Knave of Hearts", "Two", "Five", "Seven", "executioner", "Gryphon", "White roses", "Live flamingoes", "Live hedgehogs"],
    "Sea-Shore": ["Mock Turtle"],
    "Court of Justice": ["dish of tarts", "scroll of parchment"],
}
# Things inside things, and what people carry.
INSIDE_THINGS = [("ORANGE MARMALADE jar", "cupboards and book-shelves"), ("EAT ME cake", "little glass box")]
HOLD = {"White Rabbit": ["watch"], "Caterpillar": ["hookah"], "Hatter": ["hat"], "King of Hearts": ["King's crown"], "Herald": []}
HOLD.pop("Herald")
PROPS = {n: {"fixed": True} for n in ["glass table", "cupboards and book-shelves", "table under tree", "cauldron of soup", "large arm-chair", "White roses"]}
KEEP.discard("sixpence")

# Hand-written descriptions (from the book) where merges mixed several scenes together.
DESCRIBE = {
    "Riverbank": "A grassy bank by a lazy river on a hot summer afternoon. Daisies grow in the grass, a hedge runs along the field, and under the hedge is a large rabbit-hole. Alice's sister sits here reading a book with no pictures or conversations in it.",
    "The Rabbit-Hole": "The rabbit-hole goes straight on like a tunnel, then dips into a very deep well. The fall is so slow there is time to look around: the sides are lined with cupboards and book-shelves, maps and pictures hung upon pegs. At the bottom: a heap of sticks and dry leaves, and a long passage.",
    "Hall of Doors": "A long, low hall lit by a row of lamps hanging from the roof. Doors all round the hall, all locked. In the middle stands a little three-legged table of solid glass. Behind a low curtain is a little door about fifteen inches high, which leads into the loveliest garden — if only one were small enough to get through.",
    "Pool of Tears": "A salt-water pool of tears, wept by a giant Alice when she was nine feet high. Now it is deep enough to swim in, and a Mouse and many strange birds and animals have fallen into it.",
    "Shore of the Pool": "The bank of the pool of tears, where a queer-looking party of dripping birds and animals gathers to get dry: a Duck, a Dodo, a Lory, an Eaglet and others. The Dodo proposes a Caucus-race, which everybody wins.",
    "The Wood": "A thick wood of tall trees, flowers and blades of grass, with paths that lead to many odd places. The Cheshire Cat may be sitting on a bough here, grinning; a Pigeon guards her nest high in the trees.",
    "White Rabbit's House": "A neat little house with a bright brass plate on the door engraved W. RABBIT. Upstairs is a tidy little room; there is a window, a chimney, and a cucumber-frame in the garden below.",
    "Tidy Little Room": "A tidy little room upstairs in the White Rabbit's house, with a table in the window. On it lie a fan and two or three pairs of tiny white kid gloves; a little bottle stands near the looking-glass.",
    "Caterpillar's Mushroom": "A large mushroom about the same height as a little girl, among tall flowers and blades of grass. On top sits a large blue Caterpillar, arms folded, quietly smoking a long hookah. One side of the mushroom makes you grow taller, the other side shorter.",
    "Duchess's House": "A little house about four feet high, standing in an open place in the wood. A Frog-Footman sits on the ground near the door, staring stupidly up into the sky. A most extraordinary noise is going on within — howling and sneezing, and a great crash.",
    "Duchess's Kitchen": "A large kitchen full of smoke and far too much pepper. The Duchess sits on a three-legged stool nursing a sneezing baby; the cook leans over the fire, stirring a large cauldron of soup and throwing pans and dishes about. A large Cheshire cat sometimes lies on the hearth, grinning from ear to ear.",
    "March Hare's House": "A house with chimneys shaped like ears and a roof thatched with fur. In front of it, under a tree, a large table is set out for tea, and the March Hare, the Hatter and a sleeping Dormouse are crowded together at one corner of it. It is always six o'clock here, always tea-time.",
    "Tree with a Door": "One of the trees in the wood has a door leading right into it. Through it is the long hall again, close to the little glass table.",
    "The Queen's Garden": "The beautiful garden behind the little door: bright flower-beds and cool fountains. A large rose-tree near the entrance bears white roses, which three gardeners shaped like playing cards are busily painting red. Here the Queen of Hearts plays croquet with live flamingoes for mallets and live hedgehogs for balls, and orders heads off at every turn.",
    "Sea-Shore": "The sea-shore, reached with the Gryphon's help. On a little ledge of rock sits the Mock Turtle, sad and lonely, sighing as if his heart would break.",
    "Court of Justice": "The court where the King and Queen of Hearts sit on their throne, with a great crowd of little birds and beasts and the whole pack of cards. In the middle of the court is a table with a large dish of tarts. The jury-box holds twelve creatures with slates; the White Rabbit is herald, with a trumpet and a scroll of parchment.",
    "little golden key": "A tiny golden key lying on the glass table in the hall of doors. It fits the little door behind the curtain, which leads into the loveliest garden.",
    "DRINK ME bottle": "A little bottle on the glass table, with a paper label round its neck: DRINK ME, beautifully printed in large letters. It is not marked poison. It tastes of cherry-tart, custard, pine-apple, roast turkey, toffee and hot buttered toast — and whoever drinks it shuts up like a telescope, down to ten inches high.",
    "little glass box": "A little glass box lying under the glass table in the hall of doors.",
    "EAT ME cake": "A very small cake inside the little glass box, with the words EAT ME beautifully marked in currants. Whoever eats it opens out like the largest telescope that ever was — grows enormously tall.",
    "cupboards and book-shelves": "Cupboards and book-shelves lining the sides of the rabbit-hole, with maps and pictures hung upon pegs. Things can be taken from them while falling, and put back.",
    "Queen of Hearts": """The Queen of Hearts, a playing card with a crown — the ruler of Wonderland's court, small, fierce and very loud.

Personality:
- Quick to rage at the smallest thing; her one way of settling every difficulty is "Off with his head!"
- Loves croquet and must always win; cheats without noticing.
- Her orders are rarely carried out: the King quietly pardons the condemned.

Wants:
- Obedience, red roses, and her tarts back.
- Everyone to play croquet with her, at once.

Knows:
- The gardeners planted a white rose-tree by mistake.
- Someone has stolen the tarts she made, all on a summer day.

Speech:
- Shouts in short commands, stamps, turns crimson with fury; "Sentence first — verdict afterwards!\"""",
    "White Rabbit": """The White Rabbit: a rabbit with pink eyes, splendidly dressed in a waistcoat, who takes a watch out of his waistcoat-pocket and is always late. He is the Queen's herald and the Duchess's acquaintance, and he lives in a neat little house in the wood.

Personality:
- Nervous, fussy and hurried; trots along muttering to himself.
- Timid with the great (the Duchess, the Queen), bossy with servants — he takes Alice for his housemaid, Mary Ann.

Wants:
- Not to be late: "Oh my ears and whiskers, how late it's getting!"
- His fan and his white kid gloves, which he keeps dropping.

Knows:
- The way to his own house, to the Duchess, and to the Queen's croquet-ground and court.
- The Duchess is under sentence of execution for boxing the Queen's ears.

Speech:
- Anxious exclamations: "Oh dear! Oh dear! I shall be too late!", "The Duchess! Oh my dear paws! She'll get me executed, as sure as ferrets are ferrets!\"""",
    "Alice's sister": """Alice's older sister, sitting on the riverbank on a hot afternoon, reading a book that has no pictures or conversations in it.

Personality:
- Calm, patient and a little absorbed in her reading; fond of Alice.

Wants:
- To finish her chapter in peace; she does not see the White Rabbit run by.

Knows:
- Tea will be ready at home soon; their cat Dinah is waiting there.

Speech:
- Gentle, grown-up, a touch amused at Alice's questions.""",
    "ORANGE MARMALADE jar": "A jar labelled ORANGE MARMALADE, taken from one of the shelves in the rabbit-hole — to great disappointment, it is empty.",
}

missing = [n for n in KEEP | {a for v in MERGE.values() for a in v} | {a for v in MERGE_EXTRA.values() for a in v} if n not in names]
if missing:
    sys.exit(f"not in graph: {missing}")
merged_away = {a for v in MERGE.values() for a in v} | {a for v in MERGE_EXTRA.values() for a in v}
drop = sorted(n for n in names if n not in KEEP and n not in merged_away)

def q(s): return json.dumps(s, ensure_ascii=False)
out = [
    "# Manual fixes for the extracted graph of Alice's Adventures in Wonderland (Lewis Carroll, 1865;",
    "# public domain). Tied to the committed graph.json. The [drop] list is generated: everything outside",
    "# the playable map, the cast and the props kept below.",
    "# Apply (from builder/):",
    "#   uv run builder curate ../stories/alice/graph.json ../stories/alice/curation.toml -o ../stories/alice/graph.curated.json",
    "",
    "[merge]",
]
for k, v in {**MERGE, **MERGE_EXTRA}.items():
    out.append(f"{q(k)} = [{', '.join(q(a) for a in v)}]")
out += ["", "# The player is Alice (a hero in meta.toml) or their own hero; the rest is backstory, verse and scenery.", "[drop]", "entities = ["]
out += [f"    {q(n)}," for n in drop]
out += ["]", "", "[rename]"]
out += [f"{q(k)} = {q(v)}" for k, v in RENAME.items()]
out += ["", "# --- Playable map ---"]
for s, t, d in PASSAGES:
    out += ["", "[[add_relation]]", f"source = {q(s)}", f"target = {q(t)}", 'type = "LEADS_TO"', f"description = {q(d)}"]
for s, t, d in INSIDE:
    out += ["", "[[add_relation]]", f"source = {q(s)}", f"target = {q(t)}", 'type = "LOCATED_IN"', f"description = {q(d)}"]
# Pass 2: drop the extractor's placements of everything placed above (needs a pass-1 curated graph).
curated_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if curated_path and curated_path.exists():
    cg = json.loads(curated_path.read_text())
    types = {e["name"]: e["type"] for e in cg["entities"]}
    places = {n for n, t in types.items() if t in ("Spot", "Building", "Country")}
    placed = {x for v in PLACE.values() for x in v} | {a for a, _ in INSIDE_THINGS} | {x for v in HOLD.values() for x in v}
    pairs = set()
    for r in cg["relationships"]:
        a, b, t = r["source"], r["target"], r["type"]
        if t == "LEADS_TO" and a in places and b in places:
            continue
        if (a, b) in {(x, y) for x, y, _ in INSIDE}:
            continue
        if (a in places and b in places) or (a in placed and (b in places or t == "LOCATED_IN")) or (b in placed and t in ("HAS_ITEM", "HAS_AGENT")):
            if t not in ("INTERACTS_WITH", "HAS_ATTRIBUTE"):
                pairs.add((a, b, t))
    out += ["", "# --- Extracted placements replaced by the ones below ---"]
    for a, b, t in sorted(pairs):
        out += ["", "[[drop_relation]]", f"source = {q(a)}", f"target = {q(b)}", f"type = {q(t)}"]
    out += ["", "# --- Where everyone and everything starts ---"]
    for place, items in PLACE.items():
        for x in items:
            out += ["", "[[add_relation]]", f"source = {q(x)}", f"target = {q(place)}", 'type = "LOCATED_IN"']
    for a, b in INSIDE_THINGS:
        out += ["", "[[add_relation]]", f"source = {q(a)}", f"target = {q(b)}", 'type = "LOCATED_IN"']
    for who, things in HOLD.items():
        for x in things:
            out += ["", "[[add_relation]]", f"source = {q(who)}", f"target = {q(x)}", 'type = "HAS_ITEM"']
    out += ["", "[describe]"]
    out += [f"{q(k)} = {q(v)}" for k, v in DESCRIBE.items()]
    out += ["", "[props]"]
    out += [f"{q(n)} = {{ fixed = true }}" for n in PROPS]
(root / "curation.toml").write_text("\n".join(out) + "\n")
print(f"kept {len(KEEP)}, merged {len(merged_away)}, dropped {len(drop)}")
