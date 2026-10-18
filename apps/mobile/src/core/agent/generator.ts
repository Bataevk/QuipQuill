/**
 * The background world generator (stage 6.2d, ADR 42): after the master's turn, one call with a
 * structured answer reads what happened and brings the world in line with it — how people now
 * stand with the player and each other, where they went, what they became, small things the
 * narration made real or someone truthfully spoke of (gaps, ADR 45), what luck rolls brought, and
 * what the place itself implies (ADR 47). Changes go through the same world commands as the master's tools, with
 * `actor = 'generator'`; a change the world refuses is dropped, never shown to the player.
 * Successor of the legacy prototype's `updator` (docs/notes/legacy-prototype.md).
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import type { StoryMeta } from '../story/story-meta';
import {
  ADDABLE_TYPES,
  type Entity,
  eventEntityIds,
  fail,
  isBeing,
  lowerTypes,
  type Outcome,
  PLAYER_ID,
  type WorldEvent,
} from '../world/model';
import { NameBook } from '../world/names';
import type { World } from '../world/world';
import { LIKELIHOODS, SURE_LIKELIHOODS } from './luck';
import { CACHE, CACHE_LONG } from './messages';
import { brief, eventText, renderStoryStyle, renderWorldState } from './prompt';
import { invokeWithin, jsonSchemaFormat, parseJsonAnswer, strictObject } from './structured';
import type { ExpectedGap } from './middleware/guard';
import { exclusive } from './turn-lock';

export const GENERATOR_PROMPT = `You keep the world of an interactive story true to what happens in it. You do not narrate. After each turn you read what the player did, what the Game Master narrated and how people reacted, and you record the lasting consequences in the world, so that people act on what really happened rather than on what the story originally planned. You are the one who grows the world: the Game Master records the main changes during the turn and makes at most one new thing real when the moment needs it; everything else the turn narrated or implied is yours to add.

# What to record
- relations: how a person now stands with the player or with someone else, from the person's side: joined or refused a plan, betrayed, rescued, owes, fears, distrusts. If a tie the story planned is no longer true, rewrite it; if a tie has ended (someone left a crew, an alliance broke), put it in cancelled instead.
- edits: a lasting change to a person or thing: attitude, plans, what they now know, their state. Give the whole new description: keep everything that is still true and rewrite only what changed. A person whose description still says they recruited, trust or wait for the player, when the turn made that untrue, must be edited.
- moves: a person or creature the narration sent elsewhere (left, followed, fled, was taken away) who is not there yet in the world state. Destination: any location.
- additions: something the world does not have yet. where: the location or container it is in, "player" if the player now carries it, or empty for the current location. Never places, never anything powerful. Each has a basis:
  - narrated: the turn made it real. The Game Master narrated it (what the player found or got from a luck roll, what a mishap brought), or someone truthfully said it is there (the bag of tools under the bench, the service door in the alley). Add it where it was said to be. When someone lies, boasts or only guesses (their description, ties and manner tell), add nothing.
  - expected: a gap the place itself implies, which nobody had to find: a bartender behind a bar, a lamp in a lit room. likelihood: how sure you are that this place has it (certain, likely, possible, unlikely); only certain and likely are added, at most ten a turn. The input check may list gaps it expected here ("Expected here by the input check"): add those the place and the turn bear out, unless the Game Master already made them real.

# Luck
Finds and strokes of luck are never yours to invent: they come only from the Game Master's luck rolls (listed under "Luck this turn") and from what the Game Master narrated of them. A success narrated as a find is an addition with basis narrated (where "player" if it is now in their hands); a narrated mishap is recorded as whatever it changed (an edit, a relation, an addition). A failure adds nothing.

# Rules
- Record only what this turn established. Never invent new plot and never contradict the narration.
- What is listed under "Already recorded this turn" is settled: do not repeat it, and do not cancel or rewrite a tie or description recorded there.
- Leave the player character's own description alone: the Game Master keeps it. Ties to the player are yours to record.
- What a person remembers (their notes) is their own view: use it to understand them, not as fact about the world.
- Name people, things and places by their ids exactly as written in square brackets, prefix included (npc:mags-holloway, item:brass-key); the player is "player". A name instead of an id is not recorded.
- Most turns change nothing lasting: then answer with empty lists.
- Answer with the JSON object in the given shape.

# The story
{{STORY_STYLE}}`;

const ADDABLE = lowerTypes(ADDABLE_TYPES);

/** OpenAI-style `response_format` for the generator model (passed to the provider as is). */
export const GENERATOR_RESPONSE_FORMAT = jsonSchemaFormat(
  'world_changes',
  strictObject({
    relations: {
      type: 'array',
      items: strictObject({ a: { type: 'string' }, b: { type: 'string' }, description: { type: 'string' } }),
    },
    cancelled: { type: 'array', items: strictObject({ a: { type: 'string' }, b: { type: 'string' } }) },
    moves: { type: 'array', items: strictObject({ who: { type: 'string' }, to: { type: 'string' } }) },
    edits: { type: 'array', items: strictObject({ target: { type: 'string' }, description: { type: 'string' } }) },
    additions: {
      type: 'array',
      items: strictObject({
        name: { type: 'string' },
        type: { type: 'string', enum: ADDABLE },
        description: { type: 'string' },
        where: { type: 'string' },
        basis: { type: 'string', enum: ['narrated', 'expected'] },
        likelihood: { type: 'string', enum: [...LIKELIHOODS] },
      }),
    },
  }),
);

const text = z.string().trim().min(1);
/** Gaps the place implies that are added a turn at most (ADR 47, 55); only sure ones (SURE_LIKELIHOODS). */
export const MAX_EXPECTED = 10;
export const Changes = z.object({
  relations: z.array(z.object({ a: text, b: text, description: text })).default([]),
  cancelled: z.array(z.object({ a: text, b: text })).default([]),
  moves: z.array(z.object({ who: text, to: text })).default([]),
  edits: z.array(z.object({ target: text, description: text })).default([]),
  additions: z
    .array(
      z.object({
        name: text,
        type: z.enum(ADDABLE),
        description: z.string(),
        where: z.string().trim().default(''),
        basis: z.enum(['narrated', 'expected']).default('narrated'),
        likelihood: z.enum(LIKELIHOODS).default('certain'),
      }),
    )
    .default([]),
});
export type WorldChanges = z.infer<typeof Changes>;
/** The model's answer before defaults: `basis`, `likelihood` and `where` may be left out. */
export type WorldChangesInput = z.input<typeof Changes>;

/** People in the turn's context at most: those here, then those the turn touched. */
const MAX_PEOPLE = 8;
/** The latest actor notes shown for each person (ADR 45). */
const PERSON_NOTES = 5;

export interface GeneratorDeps {
  /** QueueChatModel with purpose 'generator' and GENERATOR_RESPONSE_FORMAT. */
  model: BaseChatModel;
  world: World;
  meta: Pick<StoryMeta, 'title' | 'style' | 'worldSummary'>;
  signal?: AbortSignal;
  /** Explicit limit for the call (default 30 s); the next turn waits at most this long. */
  timeoutMs?: number;
}

/** What one turn was, as the generator reads it. */
export interface GeneratorTurn {
  turn: number;
  /** What the player wrote. */
  input: string;
  /** All the master's text of the turn. */
  reply: string;
  /** Gaps the input check expected here (the Guard's recommendations, ADR 54). */
  expected?: ExpectedGap[];
}

export interface GeneratorResult {
  applied: WorldEvent[];
  /** Changes the world refused (unknown ids, broken rules) or that could not be read. */
  rejected: { change: string; error: string }[];
}

/** The model's answer is not a list of changes. */
export class GeneratorAnswerError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'GeneratorAnswerError';
  }
}

/**
 * Runs the generator for a finished turn: under the turn lock, so no turn interleaves with it
 * (the caller waits for it before the next one). Rejects if the call fails, times out, or its
 * answer cannot be read; a refused change does not fail the rest.
 */
export function generate(deps: GeneratorDeps, turn: GeneratorTurn): Promise<GeneratorResult> {
  return exclusive(deps.world, async () => {
    const answer = await invokeWithin(deps.model, await generatorMessages(deps, turn), {
      timeoutMs: deps.timeoutMs ?? 30_000,
      signal: deps.signal,
    });
    return applyChanges(deps.world, parseChanges(answer.text));
  });
}

/** System: rules and the story (stable, cached). Human: the world now and the turn. */
export async function generatorMessages(
  deps: Pick<GeneratorDeps, 'world' | 'meta'>,
  turn: GeneratorTurn,
): Promise<[SystemMessage, HumanMessage]> {
  const { world } = deps;
  const events = await world.log.ofTurn(turn.turn);
  const book = new NameBook((id) => world.state.entity(id));
  for (const event of events) for (const id of eventEntityIds(event)) await book.of(id);
  const names = book.known;

  const reactions = events.flatMap((e) =>
    e.kind === 'entity.reacted' && !e.payload.reaction.startsWith('[silence]')
      ? [`- ${names.get(e.payload.entity) ?? e.payload.entity}: ${brief(e.payload.reaction, 400)}`]
      : [],
  );
  const luck = events.flatMap((e) =>
    e.kind === 'luck.rolled' ? [`- ${e.payload.attempt} (${e.payload.stakes}): ${e.payload.outcome}`] : [],
  );
  const recorded = events
    .filter((e) => e.kind !== 'entity.reacted' && e.kind !== 'luck.rolled')
    .map((e) => `- ${eventText(e, names)}`);
  const people = await Promise.all((await involved(world, events)).map((npc) => personText(world, npc)));

  // The people first, with a breakpoint: they change less often than the world state and the
  // turn (ADR 49).
  const peopleText = `# People involved\n${people.join('\n') || '- nobody'}\n\n`;
  const context = [
    `# World state\n${await renderWorldState(world)}`,
    [
      `# This turn (${turn.turn})`,
      `Player: ${turn.input.trim()}`,
      `Game Master: ${turn.reply.trim() || '(no text)'}`,
      reactions.length ? `What people said and did:\n${reactions.join('\n')}` : '',
      luck.length ? `Luck this turn:\n${luck.join('\n')}` : '',
      turn.expected?.length
        ? `Expected here by the input check:\n${turn.expected
            .map((e) => `- ${e.name} (${e.type}, ${e.likelihood}): ${brief(e.description, 200)}`)
            .join('\n')}`
        : '',
      `Already recorded this turn:\n${recorded.join('\n') || '- nothing'}`,
    ]
      .filter(Boolean)
      .join('\n'),
  ];
  const system = GENERATOR_PROMPT.replace('{{STORY_STYLE}}', renderStoryStyle(deps.meta));
  return [
    new SystemMessage({ content: [{ type: 'text', text: system, cache_control: CACHE_LONG }] }),
    new HumanMessage({
      content: [
        { type: 'text', text: peopleText, cache_control: CACHE },
        { type: 'text', text: context.filter(Boolean).join('\n\n') },
      ],
    }),
  ];
}

export function parseChanges(answer: string): WorldChanges {
  try {
    return parseJsonAnswer(Changes, answer);
  } catch (error) {
    throw new GeneratorAnswerError('The world changes could not be read.', { cause: error });
  }
}

/** Applies the changes in a fixed order: ties first, then places, descriptions, new things. */
export async function applyChanges(world: World, changes: WorldChanges): Promise<GeneratorResult> {
  const result: GeneratorResult = { applied: [], rejected: [] };
  const apply = async (change: string, outcome: Promise<Outcome>) => {
    const done = await outcome;
    if (done.ok) result.applied.push(done.value);
    else result.rejected.push({ change, error: done.error });
  };
  await world.actingAs('generator', async () => {
    for (const c of changes.cancelled) await apply(`cancel ${c.a} — ${c.b}`, world.cancelRelation(c.a, c.b));
    for (const r of changes.relations) {
      await apply(`relation ${r.a} — ${r.b}`, world.setRelation(r.a, 'INTERACTS_WITH', r.b, r.description));
    }
    for (const m of changes.moves) {
      const who = await world.state.entity(m.who);
      await apply(
        `move ${m.who} → ${m.to}`,
        who?.type === 'PLAYER'
          ? refuse('the player moves only by their own choice')
          : world.setRelation(m.who, 'LOCATED_IN', m.to),
      );
    }
    for (const e of changes.edits) {
      await apply(
        `edit ${e.target}`,
        e.target === PLAYER_ID
          ? refuse('the master keeps the player')
          : world.setEntity({ id: e.target, description: e.description }),
      );
    }
    let expected = 0;
    for (const a of changes.additions) {
      if (a.basis === 'expected') {
        if (!SURE_LIKELIHOODS.includes(a.likelihood)) {
          await apply(`add ${a.name} (expected)`, refuse(`only certain or likely gaps are added, not ${a.likelihood}`));
          continue;
        }
        if (++expected > MAX_EXPECTED) {
          await apply(`add ${a.name} (expected)`, refuse(`at most ${MAX_EXPECTED} expected additions a turn`));
          continue;
        }
      }
      const toPlayer = a.where === PLAYER_ID;
      const where = a.where && !toPlayer ? await world.state.entity(a.where) : undefined;
      if (a.where && !toPlayer && !where) {
        await apply(`add ${a.name} in ${a.where}`, refuse(`there is no ${a.where}`));
        continue;
      }
      const before = result.applied.length;
      await apply(`add ${a.name}`, world.setEntity({ name: a.name, type: a.type, description: a.description }));
      const added = result.applied.length > before ? result.applied.at(-1) : undefined;
      if (added?.kind !== 'entity.added') continue;
      // Created here, then put where it was said to be, or into the player's hands (if that fails, it stays here).
      const id = added.payload.entity;
      if (toPlayer) await apply(`give ${a.name} to the player`, world.setRelation(PLAYER_ID, 'HAS_ITEM', id));
      else if (where) await apply(`put ${a.name} in ${where.id}`, world.setRelation(id, 'LOCATED_IN', where.id));
    }
  });
  return result;
}

/** People here, then people the turn's events name — each once, at most MAX_PEOPLE. */
async function involved(world: World, events: WorldEvent[]): Promise<Entity[]> {
  const scene = await world.scene();
  const found = new Map<string, Entity>();
  for (const e of scene.here.filter(isBeing)) found.set(e.id, e);
  for (const id of events.flatMap(eventEntityIds)) {
    if (found.has(id)) continue;
    const entity = await world.state.entity(id);
    if (entity && isBeing(entity)) found.set(id, entity);
  }
  return [...found.values()].slice(0, MAX_PEOPLE);
}

async function personText(world: World, npc: Entity): Promise<string> {
  const where = await world.state.locationOf(npc.id);
  const ties = (await world.state.ties(npc.id)).map(
    (t) => `  - ${t.other.name} [${t.other.id}]${t.description ? `: ${brief(t.description, 200)}` : ''}`,
  );
  const place = where ? `${where.name} [${where.id}]` : 'no known place';
  const notes = await world.log.actorNotes(npc.id, PERSON_NOTES);
  return [
    `- ${npc.name} [${npc.id}] (${npc.type.toLowerCase()}), in ${place}: ${brief(npc.description, 700)}`,
    ties.length ? `  Ties:\n${ties.join('\n')}` : '  Ties: none',
    notes.length ? `  Remembers:\n${notes.map((n) => `  - ${brief(n.text, 200)}`).join('\n')}` : '',
  ]
    .filter(Boolean)
    .join('\n');
}

const refuse = (why: string): Promise<Outcome> => Promise.resolve(fail(why));
