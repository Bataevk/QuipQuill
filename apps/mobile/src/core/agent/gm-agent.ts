/**
 * The game master agent: LangChain `createAgent` (ADR 25) with our model (QueueChatModel in the
 * app, a scripted model in tests), the world tools and middleware. History is ours
 * (`model_messages`, see ChatStore); no LangGraph checkpointer.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { createAgent } from 'langchain';

import type { StoryMeta } from '../story/story-meta';
import type { World } from '../world/world';
import { afterTurnMiddleware, type AfterTurnOptions } from './middleware/after-turn';
import { bugGuardMiddleware } from './middleware/bug-guard';
import { contextMiddleware, type ContextOptions } from './middleware/context';
import { guardMiddleware, type GuardOptions } from './middleware/guard';
import { invalidToolCallsMiddleware } from './middleware/invalid-tool-calls';
import { packsMiddleware } from './middleware/packs';
import { toolLimitMiddleware } from './middleware/tool-limit';
import { worldPromptMiddleware } from './middleware/world-prompt';
import type { ActorDeps } from './actor';
import { renderStoryStyle } from './prompt';
import { createGmTools, type GmToolDeps } from './tools';

export interface GmAgentOptions {
  model: BaseChatModel;
  world: World;
  meta: Pick<StoryMeta, 'title' | 'style' | 'worldSummary' | 'worldConstraints'>;
  embed?: GmToolDeps['embed'];
  /** The actor pack's model and the entities' chats, for react_entity (ADR 43, 45). */
  actor?: ActorDeps;
  /** Cheap model for the Guard (purpose 'guard'); without it player input is not screened. */
  guardModel?: BaseChatModel;
  /** Cheap model for history summaries (purpose 'summary'); without it old history is dropped. */
  summaryModel?: BaseChatModel;
  context?: Omit<ContextOptions, 'model'>;
  toolLimit?: number;
  onGuardError?: GuardOptions['onError'];
  /** Background work after a player's turn: generator and entity memory (stage 6.2i). */
  afterTurn?: Omit<AfterTurnOptions, 'world' | 'meta'>;
}

/**
 * Middleware order (agent.md): `before*` hooks run in this order, `after*` in reverse.
 * BugGuard → Guard (beforeAgent); Context (beforeModel); Packs, WorldPrompt (wrapModelCall);
 * InvalidToolCalls → ToolLimit, Packs (afterModel); AfterTurn (afterAgent, last). Packs only with an
 * actor (react_entity), AfterTurn only with background work.
 */
export function createGmAgent(options: GmAgentOptions) {
  const { model, world, meta, embed, actor } = options;
  const guard = options.guardModel
    ? [
        guardMiddleware({
          model: options.guardModel,
          world,
          worldConstraints: meta.worldConstraints,
          onError: options.onGuardError,
        }),
      ]
    : [];
  const afterTurn = options.afterTurn ? [afterTurnMiddleware({ world, meta, ...options.afterTurn })] : [];
  return createAgent({
    model,
    tools: createGmTools({ world, embed, actor }),
    middleware: [
      bugGuardMiddleware(),
      ...guard,
      contextMiddleware({ ...options.context, model: options.summaryModel }),
      toolLimitMiddleware(options.toolLimit),
      invalidToolCallsMiddleware(),
      // Packs outside WorldPrompt: the actor pack is a call of its own, not the master's prompt.
      ...(actor ? [packsMiddleware({ world, ...actor })] : []),
      worldPromptMiddleware(world, renderStoryStyle(meta)),
      // Last: a conditional spread at the head breaks createAgent's type inference.
      ...afterTurn,
    ],
  });
}

export type GmAgent = ReturnType<typeof createGmAgent>;
