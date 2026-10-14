/**
 * The game master agent: LangChain `createAgent` (ADR 25) with our model (QueueChatModel in the
 * app, a scripted model in tests), the world tools and middleware. History is ours
 * (`model_messages`, see ChatStore); no LangGraph checkpointer.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { createAgent, createMiddleware } from 'langchain';

import type { StoryMeta } from '../story/story-meta';
import type { World } from '../world/world';
import { gmSystemMessage, renderStoryStyle, renderWorldState } from './prompt';
import { createGmTools, type GmToolDeps } from './tools';

export interface GmAgentOptions {
  model: BaseChatModel;
  world: World;
  meta: Pick<StoryMeta, 'title' | 'style' | 'worldSummary'>;
  embed?: GmToolDeps['embed'];
}

export function createGmAgent({ model, world, meta, embed }: GmAgentOptions) {
  const storyStyle = renderStoryStyle(meta);
  return createAgent({
    model,
    tools: createGmTools({ world, embed }),
    middleware: [invalidToolCallsMiddleware(), worldPromptMiddleware(world, storyStyle)] as const,
  });
}

export type GmAgent = ReturnType<typeof createGmAgent>;

/**
 * [GM RULES] + [STORY STYLE] + {{WORLD_STATE}}, rebuilt from the database before every model
 * call, so the master sees the result of the tools it just ran.
 */
export function worldPromptMiddleware(world: World, storyStyle: string) {
  return createMiddleware({
    name: 'WorldPrompt',
    wrapModelCall: async (request, handler) =>
      handler({ ...request, systemMessage: gmSystemMessage(storyStyle, await renderWorldState(world)) }),
  });
}

/**
 * Tool calls whose arguments are not a JSON object land in `invalid_tool_calls`
 * (QueueChatModel), and `createAgent` does not run them: without this the turn would end
 * silently. Each one becomes a regular call (its raw arguments kept as a string) answered by an
 * error tool message, so the model sees what went wrong and can try again.
 */
export function invalidToolCallsMiddleware() {
  return createMiddleware({
    name: 'InvalidToolCalls',
    afterModel: {
      canJumpTo: ['model'],
      hook: (state) => {
        const last = state.messages.at(-1);
        if (!last || !AIMessage.isInstance(last) || !last.invalid_tool_calls?.length) return undefined;
        const invalid = last.invalid_tool_calls.map((call, i) => ({
          ...call,
          id: call.id || `invalid_${i}`,
        }));
        const replaced = new AIMessage({
          id: last.id,
          content: last.content,
          tool_calls: [
            ...(last.tool_calls ?? []),
            ...invalid.map((call) => ({
              id: call.id,
              name: call.name ?? 'unknown',
              args: { invalid_arguments: call.args ?? '' },
              type: 'tool_call' as const,
            })),
          ],
          invalid_tool_calls: [],
          usage_metadata: last.usage_metadata,
          response_metadata: last.response_metadata,
        });
        const errors = invalid.map(
          (call) =>
            new ToolMessage({
              tool_call_id: call.id,
              name: call.name ?? 'unknown',
              status: 'error',
              content:
                `Error: the arguments of ${call.name ?? 'this'} call are not a valid JSON object ` +
                `(${call.error ?? 'unparsable'}). Call it again with proper arguments.`,
            }),
        );
        // Valid calls of the same reply still run; with none, go straight back to the model.
        return last.tool_calls?.length
          ? { messages: [replaced, ...errors] }
          : { messages: [replaced, ...errors], jumpTo: 'model' as const };
      },
    },
  });
}
