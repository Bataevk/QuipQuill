import type { QueueChatModel } from '../net/queue-chat-model';
import type { Entity } from '../world/model';
import { CHARACTER_RESPONSE_FORMAT } from './character';
import { GUARD_RESPONSE_FORMAT } from './middleware/guard';
import { npcChatId } from './npc';

/**
 * The models of one playthrough over the cloud queue: one `QueueChatModel` per purpose and UI
 * channel (protocol.md). The server picks model, effort and max_tokens by purpose (ADR 10).
 */
export function queueModels(base: QueueChatModel) {
  const plain = { tools: undefined, responseFormat: undefined, meta: undefined };
  return {
    model: base.with({ ...plain, channel: 'gm', purpose: 'gm' }),
    guardModel: base.with({ ...plain, channel: 'guard', purpose: 'guard', responseFormat: GUARD_RESPONSE_FORMAT }),
    summaryModel: base.with({ ...plain, channel: 'system', purpose: 'summary' }),
    /** Character adaptation before the story (its own chat, ADR 37). */
    characterModel: base.with({
      ...plain,
      channel: 'system',
      purpose: 'character',
      responseFormat: CHARACTER_RESPONSE_FORMAT,
    }),
    /** Each NPC streams on its own channel, with its name for the speech bubble. */
    npcModel: (npc: Entity) =>
      base.with({ ...plain, channel: npcChatId(npc), purpose: 'npc', meta: { speaker: npc.name } }),
  };
}
