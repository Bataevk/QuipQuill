// foolore landing: the live demo chat, scroll reveals, example tabs and the waitlist form.
// No build step and no dependencies.

// TODO(links): the real addresses. A link left as "#" stays visible as a placeholder.
const LINKS = {
  discord: "#",
  x: "#",
  instagram: "#",
  tiktok: "#",
  telegram: "#",
  email: "", // e.g. "hello@foolore.app" — shown in the footer when set
};

// The cloud's waitlist endpoint: the same host in production (Caddy sends /waitlist/* to the
// cloud); a local preview of the site talks to the local cloud.
const LOCAL = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname) && location.port !== "8080";
const WAITLIST_URL = (LOCAL ? "http://localhost:8000" : "") + "/waitlist/site";

const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
document.documentElement.classList.add("js");

/* ---------- Links ---------- */

document.querySelectorAll("[data-link]").forEach((a) => {
  const key = a.dataset.link;
  if (key === "email") {
    if (LINKS.email) {
      a.href = `mailto:${LINKS.email}`;
      a.querySelector("[data-email]").textContent = LINKS.email;
    } else {
      a.hidden = true;
    }
    return;
  }
  a.href = LINKS[key] || "#";
  if (a.href.startsWith("http")) {
    a.target = "_blank";
    a.rel = "noopener";
  }
});

/* ---------- Sticky nav ---------- */

const nav = document.querySelector("[data-nav]");
const onScroll = () => nav.classList.toggle("is-stuck", scrollY > 40);
addEventListener("scroll", onScroll, { passive: true });
onScroll();

/* ---------- Scroll reveal ---------- */

// Siblings come in one after another.
document.querySelectorAll(".steps, .features, .plans, .timeline, .hero-copy").forEach((group) => {
  [...group.querySelectorAll(":scope > .reveal")].forEach((el, i) => el.style.setProperty("--delay", `${i * 0.09}s`));
});

const revealer = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("is-in");
      revealer.unobserve(entry.target);
    }
  },
  { rootMargin: "0px 0px -8% 0px", threshold: 0.12 },
);
document.querySelectorAll(".reveal").forEach((el) => revealer.observe(el));

// The roadmap's line draws itself when the roadmap comes into view.
const timeline = document.querySelector(".timeline");
new IntersectionObserver(
  ([entry], observer) => {
    if (!entry.isIntersecting) return;
    timeline.style.setProperty("--progress", "1");
    observer.disconnect();
  },
  { threshold: 0.3 },
).observe(timeline);

/* ---------- Hero glow follows the pointer ---------- */

const hero = document.querySelector(".hero");
const glow = document.querySelector("[data-glow]");
if (!reducedMotion && matchMedia("(pointer: fine)").matches) {
  hero.addEventListener("pointermove", (event) => {
    const box = hero.getBoundingClientRect();
    glow.style.setProperty("--gx", `${((event.clientX - box.left) / box.width) * 100}%`);
    glow.style.setProperty("--gy", `${((event.clientY - box.top) / box.height) * 100}%`);
  });
}

/* ---------- The live demo chat ---------- */

// Verbatim from a real playthrough (apps/mobile/.live-runs/alice-people1.json, docs/notes/site-content.md);
// lines may hold <em> and <br>.
const SCRIPT = [
  { you: "I ask the Dodo how I could possibly get dry." },
  { npc: "Dodo", text: "<em>Dodo rises gravely to its feet and raises one finger.</em><br>\"The fitting remedy for our present dampness is a Caucus-race.\"" },
  { gm: "The word hangs in the damp air like a hat on a peg: <em>Caucus-race.</em> The Duck stops wringing out its feathers and looks at the Dodo; the Lory, who was about to be offended at something, forgets what it was offended about; the Eaglet’s beak twitches, halfway between a sneer and a smile. Nobody says a word, because nobody is quite sure what a Caucus-race is." },
  { you: "I walk into the wood." },
  { gm: "It is a thick wood, and dim, and very still. The trees stand tall and close, their branches tangled overhead so that the light falls in green patches, like coins spilled on the moss. The grass grows high, taller than you in places, and the flowers — oh, the flowers are the odd thing. They do not wilt as you pass. They turn, very slowly, to watch you. A violet, a rose, a tall sunflower with a face like a clock, all of them openly staring, as if you were the curiosity, and not they." },
  { you: "I ask the Cheshire Cat which way I ought to go from here." },
  { npc: "Cheshire Cat", text: "<em>Cheshire Cat grins wider and settles more comfortably on the bough.</em><br>\"That depends a good deal on where you want to get to.\"" },
  { you: "I go to the March Hare’s house." },
  { gm: "The path winds on, and the wood thins, and then the trees step aside altogether — and there, in a clearing, stands the strangest house you have ever seen.<br><br>The March Hare and the Hatter are looking at each other across the sleeping Dormouse, and neither of them has noticed you yet. The teapot is in mid-air, frozen between two hands. The tea-things are everywhere, and it is always six o'clock here, always tea-time." },
];

const chat = document.querySelector("[data-hero-chat]");
const composer = document.querySelector(".composer");
const typing = document.querySelector("[data-typing]");
const MAX_LINES = 7;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function addLine(className, html) {
  const el = document.createElement("div");
  el.className = `msg ${className}`;
  el.innerHTML = html;
  chat.append(el);
  while (chat.children.length > MAX_LINES) chat.firstElementChild.remove();
  return el;
}

async function typeIntoComposer(text) {
  composer.classList.add("is-typing");
  typing.textContent = "";
  for (const ch of text) {
    typing.textContent += ch;
    await sleep(28 + Math.random() * 40);
  }
  await sleep(380);
  composer.classList.remove("is-typing");
  typing.textContent = "What do you do?";
}

// The master's text streams in a few words at a time, as in the app.
async function stream(el, text) {
  const cursor = '<span class="cursor"></span>';
  const words = text.split(" ");
  let shown = el.innerHTML;
  for (let i = 0; i < words.length; i += 1) {
    shown += (i ? " " : "") + words[i];
    el.innerHTML = shown + cursor;
    await sleep(45 + Math.random() * 70);
  }
  el.innerHTML = shown;
}

async function playScene() {
  chat.replaceChildren();
  for (const step of SCRIPT) {
    if (!chatVisible) await waitUntilVisible();
    if (step.you) {
      await typeIntoComposer(step.you);
      addLine("you", escapeHtml(step.you));
      const thinking = addLine("thinking", "<i></i><i></i><i></i>");
      await sleep(1100);
      thinking.remove();
    } else if (step.npc) {
      const el = addLine("npc", `<b>${escapeHtml(step.npc)}</b>`);
      await stream(el, step.text);
      await sleep(1300);
    } else {
      const el = addLine("gm", "");
      await stream(el, step.gm);
      await sleep(1300);
    }
  }
  await sleep(3500);
}

function escapeHtml(text) {
  return text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

// The demo runs only while it is on screen.
let chatVisible = false;
let wake = () => {};
const waitUntilVisible = () => new Promise((resolve) => (wake = resolve));
new IntersectionObserver(([entry]) => {
  chatVisible = entry.isIntersecting;
  if (chatVisible) wake();
}).observe(chat);

if (!reducedMotion) {
  (async () => {
    await sleep(900);
    for (;;) await playScene();
  })();
}

/* ---------- Example tabs ---------- */

const tabs = [...document.querySelectorAll('[role="tab"]')];

function selectTab(tab) {
  for (const other of tabs) {
    const selected = other === tab;
    other.setAttribute("aria-selected", String(selected));
    other.tabIndex = selected ? 0 : -1;
    const panel = document.getElementById(other.getAttribute("aria-controls"));
    panel.hidden = !selected;
    if (selected) {
      // Restart the messages' entrance.
      panel.querySelectorAll(".msg, .cards").forEach((el, i) => {
        el.style.animation = "none";
        void el.offsetWidth;
        el.style.animation = "";
        el.style.animationDelay = `${i * 0.18}s`;
      });
    }
  }
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    const next = tabs[(index + step + tabs.length) % tabs.length];
    next.focus();
    selectTab(next);
  });
});

/* ---------- Waitlist ---------- */

const form = document.querySelector("[data-waitlist]");
const status = document.querySelector("[data-status]");

function say(text, error = false) {
  status.textContent = text;
  status.classList.toggle("is-error", error);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = form.elements.email;
  const email = input.value.trim();
  if (!/^\S+@\S+\.\S+$/.test(email)) {
    say("That does not look like an email address.", true);
    input.focus();
    return;
  }
  const button = form.querySelector("button");
  button.disabled = true;
  say("Saving your place…");
  try {
    const response = await fetch(WAITLIST_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
      signal: AbortSignal.timeout(15000),
    });
    if (response.status === 422) throw new Error("invalid");
    if (response.status === 429) throw new Error("busy");
    if (!response.ok) throw new Error(String(response.status));
    form.hidden = true;
    say(`You are on the list. We will write to ${email} once foolore is released.`);
  } catch (error) {
    button.disabled = false;
    const reason = error instanceof Error ? error.message : "";
    say(
      reason === "invalid"
        ? "That does not look like an email address."
        : reason === "busy"
          ? "Too many tries at once — wait a minute and try again."
          : "Could not reach the server. Try again in a moment.",
      true,
    );
  }
});
