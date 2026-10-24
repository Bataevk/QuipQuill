"""Builds the site's chat excerpts verbatim from the live runs (whole paragraphs only)."""
import html, json, re, sys
from pathlib import Path

runs = Path(sys.argv[1]); site = Path(sys.argv[2]); notes = Path(sys.argv[3])
RUN = {k: {l["seq"]: l for l in json.loads((runs / f"{k}.json").read_text())["lines"]} for k in ["alice-people1", "alice-own-hero", "alice-b1"]}

def paras(run, seq, keep=None):
    ps = RUN[run][seq]["text"].split("\n\n")
    return [ps[i] for i in keep] if keep is not None else ps

def md(text):
    out = html.escape(text, quote=False)
    out = re.sub(r"\*([^*\n]+)\*", r"<em>\1</em>", out)
    return out.replace("\n", "<br>")

def gm(run, seq, keep=None): return ("gm", None, "<br><br>".join(md(p) for p in paras(run, seq, keep)), (run, seq, keep))
def you(run, seq): return ("you", None, md(RUN[run][seq]["text"]), (run, seq, None))
def npc(run, seq): return ("npc", RUN[run][seq]["name"], "<br>".join(md(p) for p in paras(run, seq)), (run, seq, None))

HERO = [
    you("alice-people1", 8), npc("alice-people1", 9), gm("alice-people1", 10, [0]),
    you("alice-people1", 11), gm("alice-people1", 12, [2]),
    you("alice-people1", 13), npc("alice-people1", 14),
    you("alice-people1", 16), gm("alice-people1", 17, [0, 4]),
]
TABS = [
    ("hero", "Your hero, fitted", "wand-sparkles",
     "Bring anyone. The story keeps who they are and fits them into its world — here, a gamer from Berlin falls into a Victorian dream.",
     [("sheet", None, "<b>Max, 17</b> — “A gamer from Berlin in a hoodie, who never puts his phone down and is sure every world has a speedrun.”", None),
      gm("alice-own-hero", 1), you("alice-own-hero", 6), gm("alice-own-hero", 7, [0, 1])]),
    ("people", "Characters with character", "drama",
     "Everyone speaks for themselves: the Dormouse is offended, the March Hare is delighted.",
     [you("alice-people1", 19), npc("alice-people1", 20), npc("alice-people1", 21), gm("alice-people1", 22, [0])]),
    ("rules", "No cheating", "shield-check",
     "Things that are not in the world do not appear — however nicely you ask.",
     [you("alice-b1", 4), gm("alice-b1", 5, [0, 2])]),
    ("memory", "It remembers", "brain",
     "A few turns earlier Alice took the key and drank from the bottle. The world kept count.",
     [you("alice-b1", 14), gm("alice-b1", 15, [1, 2, 3])]),
    ("luck", "Try your luck", "dices",
     "Daring moves roll for luck. This time the shelf gave up a jar — and a joke.",
     [you("alice-own-hero", 13), ("luck",) + gm("alice-own-hero", 14, [0, 1])[1:]]),
]

def msg_html(item, indent):
    kind, name, body = item[0], item[1], item[2]
    if kind == "sheet":
        return f'{indent}<div class="hero-sheet">{body}</div>'
    if kind == "npc":
        return f'{indent}<div class="msg npc"><b>{html.escape(name)}</b>{body}</div>'
    if kind == "luck":
        return f'{indent}<div class="msg gm"><span class="roll"><svg class="i"><use href="img/icons.svg#i-dices"/></svg> Luck roll</span>{body}</div>'
    return f'{indent}<div class="msg {kind}">{body}</div>'

# Examples section
tabs_html = []
panels = []
for i, (key, label, icon, note, items) in enumerate(TABS):
    sel = "true" if i == 0 else "false"
    extra = "" if i == 0 else ' tabindex="-1"'
    tabs_html.append(f'        <button role="tab" id="tab-{key}" aria-controls="ex-{key}" aria-selected="{sel}"{extra}>{label}</button>')
    hidden = "" if i == 0 else " hidden"
    body = "\n".join(msg_html(it, "            ") for it in items)
    panels.append(f'''        <div class="example" role="tabpanel" id="ex-{key}" aria-labelledby="tab-{key}"{hidden}>
          <div class="chat">
{body}
          </div>
          <p class="note"><svg class="i"><use href="img/icons.svg#i-{icon}"/></svg> {note}</p>
        </div>''')
examples = f'''    <section class="section examples" id="examples">
      <div class="section-head reveal">
        <p class="kicker">From real games</p>
        <h2>What a turn looks like</h2>
        <p class="sub">Every line below was written live by foolore while we played <em>Alice’s Adventures in Wonderland</em>. Only trimmed, never edited.</p>
      </div>

      <div class="tabs reveal" role="tablist" aria-label="Examples">
{chr(10).join(tabs_html)}
      </div>

      <div class="example-stage reveal">
{chr(10).join(panels)}
      </div>
    </section>
'''
index = site / "index.html"; s = index.read_text()
s = re.sub(r'    <section class="section examples" id="examples">.*?</section>\n', lambda m: examples, s, count=1, flags=re.S)
static = "\n".join(msg_html(it, "            ") for it in HERO[:3])
s = re.sub(r'(<div class="chat" data-hero-chat aria-live="off">\n            <!--[^\n]*-->\n).*?(\n          </div>\n          <div class="composer")',
           lambda m: m.group(1) + static + m.group(2), s, count=1, flags=re.S)
index.write_text(s)

# The hero animation's script
def js(item):
    kind, name, body = item[0], item[1], item[2]
    if kind == "you": return "  { you: " + json.dumps(html.unescape(re.sub("<[^>]+>", "", body)), ensure_ascii=False) + " },"
    if kind == "npc": return "  { npc: " + json.dumps(name, ensure_ascii=False) + ", text: " + json.dumps(body, ensure_ascii=False) + " },"
    return "  { gm: " + json.dumps(body, ensure_ascii=False) + " },"
main = site / "main.js"; m = main.read_text()
new_script = "// Verbatim from a real playthrough (apps/mobile/.live-runs/alice-people1.json, docs/notes/site-content.md);\n// lines may hold <em> and <br>.\nconst SCRIPT = [\n" + "\n".join(js(it) for it in HERO) + "\n];"
m = re.sub(r"// One scene of The Brass Vault.*?\n\];", lambda _: new_script, m, count=1, flags=re.S)
main.write_text(m)

# Provenance note
rows = ["# Тексты сайта из живых прогонов", "",
        "Все реплики на сайте (`site/index.html`, `SCRIPT` в `site/main.js`) взяты дословно из прогонов `alice.live.test.ts` (этап L.5), только целыми абзацами; генерирует их `site_excerpts`-скрипт из лент `apps/mobile/.live-runs/*.json` (не в git — ниже полный текст выбранных реплик).",
        "", "Прогоны 2026-10-24, облако локально с конфигом прода (Mercury 3, запасная Haiku 5.5): `alice-people1` (Алиса, путь до чаепития), `alice-b1` (Алиса, зал с дверями), `alice-own-hero` (свой герой Max).", ""]
def note(title, items):
    rows.append(f"## {title}"); rows.append("")
    for it in items:
        if it[3] is None: continue
        run, seq, keep = it[3]
        line = RUN[run][seq]
        rows.append(f"- `{run}` seq {seq} [{line['name']}]" + (f", абзацы {keep}" if keep is not None else "") + ":")
        for p in paras(run, seq, keep): rows.append("  > " + p.replace("\n", "\n  > "))
    rows.append("")
note("Анимация на главном экране", HERO)
for key, label, icon, n, items in TABS: note(f"Вкладка «{label}»", items)
notes.write_text("\n".join(rows) + "\n")
print("ok")
