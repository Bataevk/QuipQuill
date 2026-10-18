import { describe, expect, it } from 'vitest';

import { actorChannel, entityIdOf, GM_CHANNEL, isActorChannel } from './channels';

describe('channels', () => {
  it.each(['npc:mags-holloway', 'item:magic-mirror', 'doc:ledger', 'ent:statue'])(
    'an actor channel leads back to its entity: %s',
    (id) => {
      const channel = actorChannel({ id });
      expect(isActorChannel(channel)).toBe(true);
      expect(entityIdOf(channel)).toBe(id);
    },
  );

  it("an NPC's channel drops its own prefix once", () => {
    expect(actorChannel({ id: 'npc:mags' })).toBe('npc:mags');
    expect(actorChannel({ id: 'item:mirror' })).toBe('npc:item:mirror');
  });

  it('other channels are not actors', () => {
    for (const channel of [GM_CHANNEL, 'guard', 'generator', 'system', undefined]) {
      expect(isActorChannel(channel)).toBe(false);
    }
    expect(entityIdOf(GM_CHANNEL)).toBeUndefined();
  });
});
