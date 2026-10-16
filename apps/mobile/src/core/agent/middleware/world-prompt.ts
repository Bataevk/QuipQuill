import { createMiddleware } from 'langchain';

import type { World } from '../../world/world';
import { withCacheBreakpoints } from '../messages';
import { gmSystemMessage, renderWorldState, worldStateMessage } from '../prompt';

/**
 * Before every model call (ADR 46): the fixed system prompt ([GM RULES] + [STORY STYLE]), the
 * history with a cache breakpoint at its end, then the world state, rebuilt from the database —
 * so the master sees the result of the tools it just ran, and tools + system + history stay a
 * prefix the provider reads from its cache on the next call.
 */
export function worldPromptMiddleware(world: World, storyStyle: string) {
  const system = gmSystemMessage(storyStyle);
  return createMiddleware({
    name: 'WorldPrompt',
    wrapModelCall: async (request, handler) =>
      handler({
        ...request,
        systemMessage: system,
        messages: [...withCacheBreakpoints(request.messages), worldStateMessage(await renderWorldState(world))],
      }),
  });
}
