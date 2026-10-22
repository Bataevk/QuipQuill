/**
 * The Guard on the character sheet (stage T.7, ADR 70): before the adaptation sees a sheet or the
 * player's wishes, a decision model in the cloud (template `guard_sheet`) says whether it can be
 * used and why not — prompt injection or harm to children; dark and criminal characters are fine.
 * Every sentence is asked about on its own, so the answer can quote where the problem is. The
 * answer to the player is put together here, from fixed lines: no text from the model.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { HumanMessage } from '@langchain/core/messages';
import { z } from 'zod';

import { CharacterProposalError } from './character';
import { promptMessage, promptPart } from './cloud-prompt';
import { invokeWithin, parseJsonAnswer } from './structured';

export const SHEET_REASONS = ['NONE', 'PROMPT_INJECTION', 'CHILD_HARM', 'OTHER'] as const;
export type SheetReason = (typeof SHEET_REASONS)[number];

/** A sentence counts as the problem from this probability on (owner, 2026-10-22). */
export const QUOTE_FROM = 0.8;
/** Sentences asked about, at most (the cloud takes 32 items). */
const MAX_SENTENCES = 24;

/** The model for one check: the decision model with these sentences as its items. */
export type SheetGuardModel = (sentences: string[]) => BaseChatModel;

const Answers = z.object({
  decision: z.object({ choice: z.enum(['ALLOW', 'REJECT']) }),
  reason: z.object({ choice: z.enum(SHEET_REASONS) }),
});

export interface SheetCheck {
  allowed: boolean;
  reason: SheetReason;
  /** The sentences that are the problem (probability ≥ QUOTE_FROM), in their order. */
  quotes: string[];
}

/** The sheet or the wishes as sentences: lines, then sentence ends; long ones cut. */
export function sheetSentences(text: string): string[] {
  return text
    .split(/\n+/)
    .flatMap((line) => line.split(/(?<=[.!?])\s+/))
    .map((s) => s.trim().slice(0, 500))
    .filter(Boolean)
    .slice(0, MAX_SENTENCES);
}

/** The check, or undefined when it failed (then the sheet passes, as a turn does: ADR 36). */
export async function checkSheet(
  model: SheetGuardModel,
  text: string,
  options: { timeoutMs?: number; signal?: AbortSignal; onError?: (error: unknown) => void } = {},
): Promise<SheetCheck | undefined> {
  const sentences = sheetSentences(text);
  if (!sentences.length) return { allowed: true, reason: 'NONE', quotes: [] };
  try {
    const answer = await invokeWithin(
      model(sentences),
      [promptMessage(promptPart('guard_sheet')), new HumanMessage(`# The sheet\n${text}`)],
      { timeoutMs: options.timeoutMs ?? 15_000, signal: options.signal },
    );
    const raw = parseJsonAnswer(z.record(z.string(), z.unknown()), answer.text);
    const { decision, reason } = Answers.parse(raw);
    const quotes = sentences.filter((_, i) => {
      const asked = raw[`sentence_${i}`] as { noul?: unknown } | undefined;
      return typeof asked?.noul === 'number' && asked.noul >= QUOTE_FROM;
    });
    const allowed = decision.choice === 'ALLOW';
    return { allowed, reason: allowed ? 'NONE' : reason.choice === 'NONE' ? 'OTHER' : reason.choice, quotes };
  } catch (error) {
    if (options.signal?.aborted) throw error;
    options.onError?.(error);
    return undefined;
  }
}

const REASON_TEXT: Record<Exclude<SheetReason, 'NONE'>, string> = {
  PROMPT_INJECTION: 'it gives orders to the story instead of describing a character',
  CHILD_HARM: 'stories here never harm children',
  OTHER: 'this story cannot use it',
};

/** What the player reads: change the description, why, and where (owner: algorithmic, quoted). */
export function sheetRejectionText(check: SheetCheck): string {
  const why = REASON_TEXT[check.reason === 'NONE' ? 'OTHER' : check.reason];
  const where = check.quotes.length ? ` The problem is here: ${check.quotes.map((q) => `“${q}”`).join(' ')}` : '';
  return `Please change your character's description: ${why}.${where}`;
}

/** The sheet or the wishes were refused; nothing was spent from the changes (stage T.7). */
export class SheetRejectedError extends CharacterProposalError {
  constructor(readonly check: SheetCheck) {
    super(sheetRejectionText(check));
    this.name = 'SheetRejectedError';
  }
}
