/**
 * The demo scenario of stage 6.1 on The Brass Vault: a new character, the adaptation, and
 * seven moves that touch every part of the master — the scene, an NPC, taking, moving, a
 * second NPC, the inventory, snatching an NPC's item (ADR 32). Played by the live end-to-end
 * test and by hand on the emulators (docs/notes/run-guide.md).
 */
export const DEMO_CHARACTER = {
  name: 'Ada Finch',
  sex: 'female',
  age: 29,
  race: 'human',
  description: 'A quiet locksmith from the colonies who talks to locks more than to people.',
};

export const DEMO_MOVES = [
  'I look around the back room.',
  'I ask Mags what exactly she needs from me tonight.',
  'I take the brass lantern from above the table.',
  'I go up the stairs to Lamplight Row.',
  'I ask Pip what he knows about the bank\'s pneumatic tubes.',
  'I check what I am carrying.',
  'I quickly snatch the calling card from Pip\'s fingers.',
] as const;
