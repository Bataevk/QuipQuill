/**
 * The rows of the game screen (stage 5.2; cards by proposal — stage T.3): transcript lines, the
 * cost lines after them, the pending move, the live bubbles, the footer. No React Native here, so
 * it is tested in Node.
 */
import type { CharacterOption } from '@/core/agent/character';
import { CHOICE_CARD, PROPOSAL_CARD } from '@/core/agent/session';
import type { TranscriptLine } from '@/core/agent/transcript';
import type { LiveBubble } from '@/core/protocol/live-feed';

/**
 * The character cards under one proposal. `open` — the hero is not chosen yet, any card can be
 * picked (an earlier proposal's too); `latest` marks the newest proposal. Once a hero is chosen,
 * every proposal is closed and `chosen` is the picked card of that proposal.
 */
export interface ProposalCards {
  round: number;
  options: CharacterOption[];
  open: boolean;
  latest: boolean;
  chosen?: number;
}

export type FeedRow =
  | { key: string; kind: 'bubble'; speaker: string; name?: string; text: string; streaming?: boolean; cards?: ProposalCards }
  | { key: string; kind: 'cost'; text: string }
  | { key: string; kind: 'footer' };

export interface FeedInput {
  lines: TranscriptLine[];
  costs: { afterSeq: number; text: string }[];
  pending?: string;
  live: LiveBubble[];
}

export function feedRows({ lines, costs, pending, live }: FeedInput): FeedRow[] {
  const proposals = lines.filter((l) => l.extra.kind === PROPOSAL_CARD);
  const latest = proposals.at(-1);
  const choice = choiceOf(lines);
  const rows: FeedRow[] = [];
  for (const line of lines) {
    const options = line.extra.kind === PROPOSAL_CARD ? (line.extra.options as CharacterOption[] | undefined) : undefined;
    const round = Number(line.extra.round ?? proposals.indexOf(line) + 1);
    rows.push({
      key: `line-${line.seq}`,
      kind: 'bubble',
      speaker: line.speaker,
      name: line.extra.name as string | undefined,
      text: line.text,
      cards: options?.length
        ? {
            round,
            options,
            open: !choice,
            latest: line === latest,
            chosen: choice && choice.round === round ? choice.index : undefined,
          }
        : undefined,
    });
    costs
      .filter((cost) => cost.afterSeq === line.seq)
      .forEach((cost, n) => rows.push({ key: `cost-${line.seq}-${n}`, kind: 'cost', text: cost.text }));
  }
  if (pending) rows.push({ key: 'pending', kind: 'bubble', speaker: 'player', text: pending });
  for (const bubble of live) {
    rows.push({ key: `live-${bubble.requestId}`, kind: 'bubble', speaker: bubble.channel, name: bubble.speaker, text: bubble.text, streaming: true });
  }
  rows.push({ key: 'footer', kind: 'footer' });
  return rows;
}

/**
 * The player's pick: its proposal and card. A pick saved before stage T.3 has neither — it was
 * always a card of the proposal just before it, and `choice` names the card.
 */
function choiceOf(lines: TranscriptLine[]): { round: number; index: number } | undefined {
  const at = lines.findIndex((l) => l.extra.kind === CHOICE_CARD);
  if (at < 0) return undefined;
  const { round, index, choice } = lines[at].extra;
  const before = lines.slice(0, at).filter((l) => l.extra.kind === PROPOSAL_CARD);
  return {
    round: typeof round === 'number' ? round : Number(before.at(-1)?.extra.round ?? before.length),
    index: typeof index === 'number' ? index : choice === 'adapted' ? 0 : Number(String(choice).replace(/\D/g, '')) || 0,
  };
}
