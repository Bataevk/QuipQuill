/**
 * Spike 3.1b: LangChain JS v1 `createAgent` with our middleware on Hermes.
 *
 * Checks what the game master needs from the framework:
 * - a custom chat model (subclass of BaseChatModel) — the future QueueChatModel;
 * - one tool with a zod schema that changes state;
 * - dynamicSystemPromptMiddleware rebuilding {{WORLD_STATE}} before every model call;
 * - a custom beforeAgent hook that can short-circuit the run (Guard → reject);
 * - a tool error returned as text, not thrown.
 *
 * Pure TypeScript: runs in Node (`node spikes/run-node.ts`) and in the app (spike screen).
 */
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, type BaseMessage, SystemMessage } from '@langchain/core/messages';
import type { ChatResult } from '@langchain/core/outputs';
import { createAgent, createMiddleware, dynamicSystemPromptMiddleware, tool } from 'langchain';
import { z } from 'zod';

type Step = { text?: string; toolCalls?: { name: string; args: Record<string, unknown> }[] };

/** Answers from a script and remembers the prompts it was given. */
class ScriptedChatModel extends BaseChatModel {
  readonly seen: BaseMessage[][] = [];
  private readonly script: Step[];
  private step = 0;

  constructor(script: Step[]) {
    super({});
    this.script = script;
  }

  _llmType(): string {
    return 'scripted';
  }

  // createAgent binds tools; the model itself ignores them (the script decides).
  bindTools(): this {
    return this;
  }

  async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    this.seen.push(messages);
    const next = this.script[Math.min(this.step, this.script.length - 1)];
    this.step += 1;
    const message = new AIMessage({
      content: next.text ?? '',
      tool_calls: (next.toolCalls ?? []).map((call, i) => ({
        id: `call_${this.step}_${i}`,
        name: call.name,
        args: call.args,
        type: 'tool_call' as const,
      })),
    });
    return { generations: [{ text: next.text ?? '', message }] };
  }
}

export type AgentSpikeResult = {
  ok: boolean;
  checks: Record<string, boolean>;
  ms: { firstRun: number; secondRun: number };
  error?: string;
};

function buildWorld() {
  const world = { location: 'loc:hall', exits: { 'loc:hall': ['loc:corridor'] } as Record<string, string[]> };

  const move = tool(
    async ({ to }: { to: string }) => {
      const exits = world.exits[world.location] ?? [];
      if (!exits.includes(to)) {
        return `Error: there is no path from ${world.location} to ${to}. Exits: ${exits.join(', ')}`;
      }
      world.location = to;
      return `Moved to ${to}.`;
    },
    {
      name: 'move',
      description: 'Move the player to an adjacent location.',
      schema: z.object({ to: z.string().describe('Location id') }),
    },
  );

  const worldState = dynamicSystemPromptMiddleware(
    () => `GM RULES\n[WORLD STATE]\nLocation: ${world.location}`,
  );

  const guard = createMiddleware({
    name: 'Guard',
    beforeAgent: {
      canJumpTo: ['end'],
      hook: (state) => {
        const last = state.messages.at(-1);
        const text = typeof last?.content === 'string' ? last.content : '';
        if (text.includes('I cast fire from my fingers')) {
          return {
            messages: [new AIMessage('Your fingers stay cold: nothing in this world answers.')],
            jumpTo: 'end',
          };
        }
        return undefined;
      },
    },
  });

  return { world, tools: [move], middleware: [guard, worldState] as const };
}

export async function runAgentSpike(): Promise<AgentSpikeResult> {
  const checks: Record<string, boolean> = {};
  const ms = { firstRun: -1, secondRun: -1 };
  try {
    // Turn 1: the model moves (one bad call first, to check text errors), then narrates.
    const { world, tools, middleware } = buildWorld();
    const model = new ScriptedChatModel([
      { toolCalls: [{ name: 'move', args: { to: 'loc:vault' } }] },
      { toolCalls: [{ name: 'move', args: { to: 'loc:corridor' } }] },
      { text: 'You step into the torchlit corridor.' },
    ]);
    const agent = createAgent({ model, tools, middleware });

    let started = Date.now();
    const result = await agent.invoke({ messages: [{ role: 'user', content: 'I walk north.' }] });
    ms.firstRun = Date.now() - started;

    const messages = result.messages;
    const toolTexts = messages.filter((m) => m.getType() === 'tool').map((m) => String(m.content));
    checks.toolErrorAsText = toolTexts[0]?.startsWith('Error: there is no path') ?? false;
    checks.toolChangedState = world.location === 'loc:corridor';
    checks.finalAnswer = String(messages.at(-1)?.content) === 'You step into the torchlit corridor.';
    checks.threeModelCalls = model.seen.length === 3;
    const systemPrompts = model.seen.map((call) =>
      call[0] instanceof SystemMessage ? String(call[0].content) : '',
    );
    checks.promptRebuiltPerCall =
      systemPrompts[0].endsWith('loc:hall') && systemPrompts[2].endsWith('loc:corridor');

    // Turn 2: Guard rejects; the model must not be called.
    const guarded = buildWorld();
    const silent = new ScriptedChatModel([{ text: 'must not be called' }]);
    const guardedAgent = createAgent({ model: silent, tools: guarded.tools, middleware: guarded.middleware });
    started = Date.now();
    const rejected = await guardedAgent.invoke({
      messages: [{ role: 'user', content: 'I cast fire from my fingers' }],
    });
    ms.secondRun = Date.now() - started;
    checks.guardShortCircuits =
      silent.seen.length === 0 && String(rejected.messages.at(-1)?.content).startsWith('Your fingers');

    return { ok: Object.values(checks).every(Boolean), checks, ms };
  } catch (error) {
    const e = error as Error;
    return { ok: false, checks, ms, error: `${e.name}: ${e.message}\n${e.stack ?? ''}`.slice(0, 2000) };
  }
}
