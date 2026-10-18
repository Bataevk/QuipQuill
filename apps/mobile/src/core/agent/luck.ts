/**
 * Luck (ADR 47): when the outcome of what the player tries is genuinely uncertain — a search, a
 * lucky break — the master asks for a roll and the code rolls, not the model (models are poor
 * at chance and play along with the player). The master names how likely it is and what is at
 * stake, in words; the story's luck (`meta.luck`) moves the odds, the more the higher the stakes.
 * Only a very unlucky world turns a failure into a mishap. What the outcome is, the master
 * decides; the generator records it in the world after his reply.
 */

/** How likely the attempt is to work in this place, as the master judges it. */
export const LIKELIHOODS = ['certain', 'likely', 'possible', 'unlikely'] as const;
export type Likelihood = (typeof LIKELIHOODS)[number];

/** What the player stands to gain. */
export const STAKES = ['trivial', 'useful', 'valuable'] as const;
export type Stakes = (typeof STAKES)[number];

export type LuckOutcome = 'success' | 'failure' | 'mishap';

/** The chance of each likelihood in a fair world (luck 0.5). */
const BASE: Record<Likelihood, number> = { certain: 0.95, likely: 0.75, possible: 0.45, unlikely: 0.15 };
/** How much luck moves the odds, by stakes: a trifle is found almost anyway, a treasure is luck. */
const WEIGHT: Record<Stakes, number> = { trivial: 0.3, useful: 0.6, valuable: 1 };
/** Below this luck a failure may become a mishap: every failure at luck 0. */
export const MISHAP_BELOW = 0.25;

export interface LuckRoll {
  attempt: string;
  likelihood: Likelihood;
  stakes: Stakes;
  luck: number;
  chance: number;
  roll: number;
  outcome: LuckOutcome;
}

/** The chance of success: the base for the likelihood, moved by luck as far as the stakes allow. */
export function luckChance(likelihood: Likelihood, stakes: Stakes, luck: number): number {
  const chance = BASE[likelihood] * (1 + WEIGHT[stakes] * (luck - 0.5) * 2);
  return Math.min(0.98, Math.max(0.02, chance));
}

/** One roll; `random` returns [0, 1) (Math.random in the app, fixed values in tests). */
export function rollLuck(
  input: { attempt: string; likelihood: Likelihood; stakes: Stakes },
  luck: number,
  random: () => number,
): LuckRoll {
  const chance = luckChance(input.likelihood, input.stakes, luck);
  const roll = random();
  let outcome: LuckOutcome = roll < chance ? 'success' : 'failure';
  if (outcome === 'failure' && luck < MISHAP_BELOW && random() < (MISHAP_BELOW - luck) / MISHAP_BELOW) {
    outcome = 'mishap';
  }
  return { ...input, luck, chance: round(chance), roll: round(roll), outcome };
}

const round = (n: number) => Math.round(n * 1000) / 1000;
