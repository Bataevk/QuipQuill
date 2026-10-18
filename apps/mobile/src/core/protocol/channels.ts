/**
 * Channels of the cloud stream (protocol.md): where the UI draws a generation. The master
 * narrates on 'gm'; each entity speaks on its own 'npc:<id>', which is also its chat id
 * (`model_messages`) and its `transcript` speaker. Background work has channels of its own
 * ('guard', 'generator', 'system').
 */
export const GM_CHANNEL = 'gm';

const ACTOR_PREFIX = 'npc:';

/** An entity's channel: 'npc:<id>', where an NPC's id loses its own 'npc:' ('npc:mags' → 'npc:mags'). */
export const actorChannel = (entity: { id: string }) => `${ACTOR_PREFIX}${entity.id.replace(/^npc:/, '')}`;

export const isActorChannel = (channel: string | undefined): boolean => channel?.startsWith(ACTOR_PREFIX) ?? false;

/**
 * The entity an actor channel belongs to, the inverse of `actorChannel` (story and game ids are
 * prefixed: 'npc:item:mirror' → 'item:mirror', 'npc:mags' → 'npc:mags'); undefined for others.
 */
export function entityIdOf(channel: string): string | undefined {
  if (!isActorChannel(channel)) return undefined;
  const rest = channel.slice(ACTOR_PREFIX.length);
  return rest.includes(':') ? rest : `${ACTOR_PREFIX}${rest}`;
}
