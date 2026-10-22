import type { QueueChatModel } from '../net/queue-chat-model';
import type { Entity } from '../world/model';
import { actorChannel, GM_CHANNEL } from '../protocol/channels';
import { CHARACTER_RESPONSE_FORMAT } from './character';
import { NOTE_ENTITY_RESPONSE_FORMAT } from './entity-memory';
import { GENERATOR_RESPONSE_FORMAT } from './generator';
import { PREGEN_RESPONSE_FORMAT } from './middleware/before-generator';

/**
 * The models of one playthrough over the cloud queue: one `QueueChatModel` per purpose and UI
 * channel (protocol.md). The server picks model, effort and max_tokens by purpose (ADR 10).
 */
export function queueModels(base: QueueChatModel) {
  const plain = { tools: undefined, responseFormat: undefined, meta: undefined };
  return {
    model: base.with({ ...plain, channel: GM_CHANNEL, purpose: 'gm' }),
    /** The input check: a decision model in the cloud, answers as JSON (ADR 70). */
    guardModel: base.with({ ...plain, channel: 'guard', purpose: 'guard' }),
    /** The input check on a sheet: each sentence asked about on its own (stage T.7, ADR 70). */
    sheetGuardModel: (sentences: string[]) => base.with({ ...plain, channel: 'guard', purpose: 'guard', items: sentences }),
    /** The answer to a rejected input, streamed as the master's (stage T.6). */
    miniGmModel: base.with({ ...plain, channel: GM_CHANNEL, purpose: 'mini_gm' }),
    /** Before the master: drafts of the place, the player's claims (stage T.8). Not a story channel. */
    pregenModel: base.with({ ...plain, channel: 'system', purpose: 'pregen', responseFormat: PREGEN_RESPONSE_FORMAT }),
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
    /** Entities' notes from their overflowing chats (ADR 45): background, not a story channel. */
    memoryModel: base.with({
      ...plain,
      channel: 'system',
      purpose: 'summary',
      responseFormat: NOTE_ENTITY_RESPONSE_FORMAT,
    }),
    /** The actor pack: each entity streams on its own channel, with its name for the bubble. */
    actorModel: (entity: Entity) =>
      base.with({ ...plain, channel: actorChannel(entity), purpose: 'npc', meta: { speaker: entity.name } }),
  };
}
