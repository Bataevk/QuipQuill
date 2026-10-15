import { createMiddleware } from 'langchain';

import type { World } from '../../world/world';
import { gmSystemMessage, renderWorldState } from '../prompt';

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
