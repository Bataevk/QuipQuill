/**
 * The background world generator (stage 6.2d, ADR 42): after the master's turn, one call with a
 * structured answer reads what happened and brings the world in line with it — how people now
 * stand with the player and each other, where they went, what they became, small things the
 * narration made real. Changes go through the same world commands as the master's tools, with
 * `actor = 'generator'`; a change the world refuses is dropped, never shown to the player.
 * Successor of the legacy prototype's `updator` (docs/notes/legacy-prototype.md).
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import type { StoryMeta } from '../story/story-meta';
import { fail, PLAYER_ID, type Entity, type EntityType, type Outcome, type WorldEvent } from '../world/model';
import type { World } from '../world/world';
import { brief, eventText, renderStoryStyle, renderWorldState } from './prompt';
import { exclusive } from './turn-lock';

export const GENERATOR_PROMPT = `You keep the world of an interactive story true to what happens in it. You do not narrate. After each turn you read what the player did, what the Game Master narrated and how people reacted, and you record the lasting consequences in the world, so that people act on what really happened rather than on what the story originally planned.

# What to record
- relations: how a person now stands with the player or with someone else, from the person's side — joined or refused a plan, betrayed, rescued, owes, fears, distrusts. If a tie the story planned is no longer true, rewrite it; if a tie has ended (someone left a crew, an alliance broke), put it in cancelled instead.
- edits: a lasting change to a person or thing — attitude, plans, what they now know, their state. Give the whole new description: keep everything that is still true and rewrite only what changed. A person whose description still says they recruited, trust or wait for the player when the turn made that untrue must be edited.
- moves: a person or creature the narration sent elsewhere (left, followed, fled, was taken away) who is not there yet in the world state. Destination: any location by id.
- additions: something the narration made real in the current location that the world does not have yet — a small thing, a passer-by. Never places, never anything powerful.

# Rules
- Record only what this turn established. Never invent new plot, never contradict the narration, never undo what is already recorded.
- Do not repeat changes listed under "Already recorded this turn".
- Leave the player character alone: the Game Master keeps their description.
- Refer to people, things and places by the ids in square brackets.
- Most turns change nothing lasting: then answer with empty lists.
- Answer with JSON only, in the given shape.

# The story
{{STORY_STYLE}}`;

/** OpenAI-style `response_format` for the generator model (passed to the provider as is). */
export const GENERATOR_RESPONSE_FORMAT = {
  type: 'json_schema',
  json_schema: {
    name: 'world_changes',
    strict: true,
    schema: {
      type: 'object',
      properties: {
        relations: {
          type: 'array',
          items: obj({ a: { type: 'string' }, b: { type: 'string' }, description: { type: 'string' } }),
        },
        cancelled: { type: 'array', items: obj({ a: { type: 'string' }, b: { type: 'string' } }) },
        moves: { type: 'array', items: obj({ who: { type: 'string' }, to: { type: 'string' } }) },
        edits: { type: 'array', items: obj({ target: { type: 'string' }, description: { type: 'string' } }) },
        additions: {
          type: 'array',
          items: obj({
            name: { type: 'string' },
            type: { type: 'string', enum: ['item', 'document', 'npc', 'creature', 'other'] }, // ADDABLE
            description: { type: 'string' },
          }),
        },
      },
      required: ['relations', 'cancelled', 'moves', 'edits', 'additions'],
      additionalProperties: false,
    },
  },
} as const;

function obj<P extends Record<string, unknown>>(properties: P) {
  return { type: 'object', properties, required: Object.keys(properties), additionalProperties: false } as const;
}

const text = z.string().trim().min(1);
const ADDABLE = ['item', 'document', 'npc', 'creature', 'other'] as const;
const Changes = z.object({
  relations: z.array(z.object({ a: text, b: text, description: text })).default([]),
  cancelled: z.array(z.object({ a: text, b: text })).default([]),
  moves: z.array(z.object({ who: text, to: text })).default([]),
  edits: z.array(z.object({ target: text, description: text })).default([]),
  additions: z
    .array(z.object({ name: text, type: z.enum(ADDABLE), description: z.string() }))
    .default([]),
});
export type WorldChanges = z.infer<typeof Changes>;

/** People in the turn's context at most: those here, then those the turn touched. */
const MAX_PEOPLE = 8;
const BEINGS: readonly EntityType[] = ['NPC', 'CREATURE'];

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
    const timeout = AbortSignal.timeout(deps.timeoutMs ?? 30_000);
    const answer = await deps.model.invoke(await generatorMessages(deps, turn), {
      signal: deps.signal ? AbortSignal.any([deps.signal, timeout]) : timeout,
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
  const names = new Map<string, string>();
  const name = async (id: string) => {
    if (!names.has(id)) names.set(id, (await world.state.entity(id))?.name ?? id);
    return names.get(id)!;
  };
  for (const event of events) for (const id of idsOf(event)) await name(id);

  const reactions = events.flatMap((e) =>
    e.kind === 'npc.reacted' && !e.payload.reaction.startsWith('[silence]')
      ? [`- ${names.get(e.payload.npc) ?? e.payload.npc}: ${brief(e.payload.reaction, 400)}`]
      : [],
  );
  const recorded = events.filter((e) => e.kind !== 'npc.reacted').map((e) => `- ${eventText(e, names)}`);
  const people = await Promise.all((await involved(world, events)).map((npc) => personText(world, npc)));

  const context = [
    `# World state\n${await renderWorldState(world)}`,
    people.length ? `# People involved\n${people.join('\n')}` : '',
    [
      `# This turn (${turn.turn})`,
      `Player: ${turn.input.trim()}`,
      `Game Master: ${turn.reply.trim() || '(no text)'}`,
      reactions.length ? `Reactions:\n${reactions.join('\n')}` : '',
      `Already recorded this turn:\n${recorded.join('\n') || '- nothing'}`,
    ]
      .filter(Boolean)
      .join('\n'),
  ];
  const system = GENERATOR_PROMPT.replace('{{STORY_STYLE}}', renderStoryStyle(deps.meta));
  return [
    new SystemMessage({ content: [{ type: 'text', text: system, cache_control: { type: 'ephemeral' } }] }),
    new HumanMessage(context.filter(Boolean).join('\n\n')),
  ];
}

export function parseChanges(answer: string): WorldChanges {
  try {
    return Changes.parse(JSON.parse(answer.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '')));
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
    for (const a of changes.additions) {
      await apply(`add ${a.name}`, world.setEntity({ name: a.name, type: a.type, description: a.description }));
    }
  });
  return result;
}

/** People here, then people the turn's events name — each once, at most MAX_PEOPLE. */
async function involved(world: World, events: WorldEvent[]): Promise<Entity[]> {
  const scene = await world.scene();
  const found = new Map<string, Entity>();
  for (const e of scene.here.filter((x) => BEINGS.includes(x.type))) found.set(e.id, e);
  for (const id of events.flatMap(idsOf)) {
    if (found.has(id)) continue;
    const entity = await world.state.entity(id);
    if (entity && BEINGS.includes(entity.type)) found.set(id, entity);
  }
  return [...found.values()].slice(0, MAX_PEOPLE);
}

async function personText(world: World, npc: Entity): Promise<string> {
  const where = await world.state.locationOf(npc.id);
  const ties = (await world.state.ties(npc.id)).map(
    (t) => `  - ${t.other.name} [${t.other.id}]${t.description ? `: ${brief(t.description, 200)}` : ''}`,
  );
  const place = where ? `${where.name} [${where.id}]` : 'no known place';
  return [
    `- ${npc.name} [${npc.id}] (${npc.type.toLowerCase()}), in ${place}: ${brief(npc.description, 700)}`,
    ties.length ? `  Ties:\n${ties.join('\n')}` : '  Ties: none',
  ].join('\n');
}

function idsOf(event: WorldEvent): string[] {
  const p = event.payload as Record<string, unknown>;
  return ['npc', 'id', 'entity', 'src', 'dst', 'a', 'b', 'item', 'from', 'to']
    .map((key) => p[key])
    .filter((v): v is string => typeof v === 'string' && v.includes(':'));
}

const refuse = (why: string): Promise<Outcome> => Promise.resolve(fail(why));
