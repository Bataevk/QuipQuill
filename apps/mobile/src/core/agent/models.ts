import type { QueueChatModel } from '../net/queue-chat-model';
import type { Entity } from '../world/model';
import { actorChannel } from './actor';
import { CHARACTER_RESPONSE_FORMAT } from './character';
import { GENERATOR_RESPONSE_FORMAT } from './generator';
import { GUARD_RESPONSE_FORMAT } from './middleware/guard';

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
    /** The background world generator after each turn (stage 6.2d): not a story channel. */
    generatorModel: base.with({
      ...plain,
      channel: 'generator',
      purpose: 'generator',
      responseFormat: GENERATOR_RESPONSE_FORMAT,
    }),
    /** Actors (react_entity): the inner voice is for the master, so not a story channel. */
    actorModel: base.with({ ...plain, channel: 'actor', purpose: 'npc' }),
    /** The actor pack: the entity's line on its own channel, with its name for the bubble. */
    actorLineModel: (entity: Entity) =>
      base.with({ ...plain, channel: actorChannel(entity), purpose: 'npc', meta: { speaker: entity.name } }),
  };
}
