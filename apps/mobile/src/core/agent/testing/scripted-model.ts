/**
 * A chat model that plays a script instead of calling an LLM: the test bench for the agent
 * (stage 4.1). Each model call takes the next step; a step may look at the messages it got
 * (e.g. the last tool result) to decide. Records every call for assertions.
 */
import { BaseChatModel, type BindToolsInput } from '@langchain/core/language_models/chat_models';
import { AIMessage, type BaseMessage } from '@langchain/core/messages';
import type { ChatResult } from '@langchain/core/outputs';

export interface ScriptStep {
  text?: string;
  toolCalls?: { name: string; args: Record<string, unknown> }[];
  /** Tool calls whose arguments failed to parse (as QueueChatModel reports them). */
  invalidToolCalls?: { name: string; args: string }[];
}

export type Step = ScriptStep | ((messages: BaseMessage[]) => ScriptStep);

export class ScriptedChatModel extends BaseChatModel {
  /** The messages of every call, system prompt first. */
  readonly calls: BaseMessage[][] = [];
  /** Tool names bound by the agent. */
  boundTools: string[] = [];
  private readonly script: Step[];
  private step = 0;

  constructor(script: Step[]) {
    super({});
    this.script = script;
  }

  _llmType(): string {
    return 'scripted';
  }

  /** Adds steps for the next turn. */
  push(...steps: Step[]): void {
    this.script.push(...steps);
  }

  get remaining(): number {
    return this.script.length - this.step;
  }

  bindTools(tools: BindToolsInput[]): this {
    this.boundTools = tools.map((t) => (t as { name?: string }).name ?? '?');
    return this;
  }

  async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    this.calls.push(messages);
    const next = this.script[this.step];
    if (!next) throw new Error(`The script has no step ${this.step + 1}`);
    this.step += 1;
    const step = typeof next === 'function' ? next(messages) : next;
    const n = this.step;
    const message = new AIMessage({
      content: step.text ?? '',
      tool_calls: (step.toolCalls ?? []).map((call, i) => ({
        id: `call_${n}_${i}`,
        name: call.name,
        args: call.args,
        type: 'tool_call' as const,
      })),
      invalid_tool_calls: (step.invalidToolCalls ?? []).map((call, i) => ({
        id: `bad_${n}_${i}`,
        name: call.name,
        args: call.args,
        error: 'Arguments are not a JSON object',
        type: 'invalid_tool_call' as const,
      })),
    });
    return { generations: [{ text: step.text ?? '', message }] };
  }
}
