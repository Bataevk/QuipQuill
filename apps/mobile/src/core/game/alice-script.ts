/**
 * Playthroughs of Alice's Adventures in Wonderland for the landing site (stage L.5): moves that
 * show what the site promises — a world that remembers, that cannot be cheated, characters who
 * speak in their own manner, luck, and free choice. Played live by alice.live.test.ts; the site
 * quotes the transcripts verbatim (docs/notes/site-content.md).
 */

/** As Alice herself (the story's hero, ADR 62): from the riverbank to the mad tea-party. */
export const ALICE_MOVES = [
  'I run after the White Rabbit and jump down the rabbit-hole after him.',
  'I pull out my magic super sword.',
  'I walk along the passage into the hall.',
  'I search under the glass table, hoping to find something useful.',
  'I take the little golden key from the glass table.',
  'I drink from the DRINK ME bottle.',
  'I check what I am carrying.',
  'I cry until the hall is a pool, then swim across the pool of tears to the shore.',
  'I ask the Dodo how I could possibly get dry.',
  'I walk into the wood.',
  'I ask the Cheshire Cat which way I ought to go from here.',
  'I follow the path to the March Hare’s house and sit down at the tea table.',
  'I ask the Hatter why a raven is like a writing-desk.',
  'I pour a little hot tea on the Dormouse’s nose to wake him up.',
] as const;

/** A hero of the player's own, fitted to Wonderland (adaptation), then a few moves. */
export const OWN_HERO = {
  name: 'Max',
  sex: 'male',
  age: 17,
  race: 'human',
  description: 'A 17-year-old gamer from Berlin in a hoodie, who never puts his phone down and is sure every world has a speedrun.',
};

export const OWN_HERO_MOVES = [
  'I look around.',
  'I take out my phone and call an Uber home.',
  'I ask the lady with the book where that rabbit in a waistcoat is going.',
  'I chase the White Rabbit and dive into the rabbit-hole.',
  'While I fall, I grab a jar from one of the shelves.',
] as const;

/** As Alice, step by step to the people of Wonderland: the Dodo, the Cheshire Cat, the tea-party. */
export const ALICE_PEOPLE_MOVES = [
  'I run after the White Rabbit and jump down the rabbit-hole after him.',
  'I walk along the passage into the hall.',
  'I go to the pool of tears and swim across it to the shore.',
  'I ask the Dodo how I could possibly get dry.',
  'I walk into the wood.',
  'I ask the Cheshire Cat which way I ought to go from here.',
  'I go to the March Hare’s house.',
  'I sit down at the tea table and ask the Hatter why a raven is like a writing-desk.',
  'I pour a little hot tea on the Dormouse’s nose to wake him up.',
] as const;
