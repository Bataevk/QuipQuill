import { useCallback, useEffect, useRef, useState } from 'react';

import type { CharacterProposal } from '@/core/agent/character';
import type { TranscriptLine } from '@/core/agent/transcript';
import type { Game, GamePhase, GameStep } from '@/core/game/game';
import { LiveFeed, type LiveBubble } from '@/core/protocol/live-feed';
import { formatUsage } from '@/core/protocol/usage-meter';
import { getCloud, openGame } from '@/platform/services';

export type { LiveBubble };

export interface GameView {
  ready: boolean;
  phase: GamePhase | undefined;
  lines: TranscriptLine[];
  live: LiveBubble[];
  /** What the player just sent, shown until the transcript has it. */
  pending: string | undefined;
  proposal: CharacterProposal | undefined;
  busy: boolean;
  /** A generation is queued or thinking and no text has come yet. */
  thinking: boolean;
  error: string | undefined;
  /** The game cannot go on by itself (`Game.nextStep()`): show "continue". */
  stalled: boolean;
  /** The cloud connection is up; while it is not, requests wait and are sent on reconnect. */
  online: boolean;
  /** Dev builds only (stage 6.1): what each action cost, shown after the line it ended on. */
  costs: { afterSeq: number; text: string }[];
  /** Dev builds only: what this game cost since the screen opened. */
  sessionCost: string | undefined;
  send(text: string): void;
  clarify(text: string): void;
  choose(index: number): void;
  recover(): void;
}

interface Snapshot {
  phase: GamePhase;
  lines: TranscriptLine[];
  proposal: CharacterProposal | undefined;
  step: GameStep;
}

async function snapshot(game: Game): Promise<Snapshot> {
  const [phase, lines, proposal, step] = await Promise.all([
    game.phase(),
    game.lines(),
    game.proposal(),
    game.nextStep(),
  ]);
  return { phase, lines, proposal: phase === 'adapting' ? proposal : undefined, step };
}

/**
 * The game screen's state over one `Game`: committed lines from the transcript, live bubbles
 * (LiveFeed) while an action runs, one action at a time. On open it finishes what needs no
 * decision (`start`, `open`); an unfinished turn waits for the player. Leaving the screen aborts
 * a running action and closes the save when it settles.
 */
export function useGame(saveId: string): GameView {
  const gameRef = useRef<Game | undefined>(undefined);
  const feed = useRef(new LiveFeed());
  const abortRef = useRef<AbortController | undefined>(undefined);
  const runningRef = useRef<Promise<unknown> | undefined>(undefined);
  const [ready, setReady] = useState(false);
  const [state, setState] = useState<Snapshot | undefined>();
  const [live, setLive] = useState<LiveBubble[]>([]);
  const [pending, setPending] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string>();
  const [online, setOnline] = useState(() => getCloud().socket.status === 'ready');
  const [costs, setCosts] = useState<{ afterSeq: number; text: string }[]>([]);
  const [sessionCost, setSessionCost] = useState<string>();

  useEffect(() => getCloud().socket.onStatus((status) => setOnline(status === 'ready')), []);

  const run = useCallback((action: (game: Game, signal: AbortSignal) => Promise<unknown>, shown?: string) => {
    const game = gameRef.current;
    if (!game || runningRef.current) return;
    const controller = new AbortController();
    abortRef.current = controller;
    feed.current.begin();
    setLive([]);
    setBusy(true);
    setError(undefined);
    setPending(shown);
    let failure: string | undefined;
    const work = action(game, controller.signal)
      .catch((e: unknown) => {
        failure = e instanceof Error ? e.message : String(e);
      })
      .then(() => (controller.signal.aborted ? undefined : snapshot(game)))
      .then((next) => {
        if (controller.signal.aborted) return;
        // One synchronous update: the transcript takes over from the live bubbles in a single
        // render, so nothing shows twice or flickers.
        feed.current.end();
        if (next) setState(next);
        if (__DEV__ && game.lastUsage && next) {
          const afterSeq = next.lines.at(-1)?.seq ?? 0;
          const line = formatUsage(game.lastUsage);
          setCosts((list) => [...list, { afterSeq, text: line }]);
          setSessionCost(formatUsage(game.sessionUsage));
        }
        setLive([]);
        setPending(undefined);
        setThinking(false);
        setError(failure);
        setBusy(false);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => {
        runningRef.current = undefined;
      });
    runningRef.current = work;
  }, []);

  useEffect(() => {
    let closed = false;
    let off: (() => void) | undefined;
    const liveFeed = feed.current;
    openGame(saveId)
      .then(async (game) => {
        if (closed) return game.close();
        gameRef.current = game;
        off = game.onStream((event) => {
          if (event.type === 'status') return setThinking(true);
          const change = feed.current.apply(event);
          if (!change) return;
          setThinking(false);
          setLive(feed.current.bubbles());
          if (change === 'new' && 'requestId' in event && event.requestId) {
            const requestId = event.requestId;
            if (!feed.current.speakerKnown(requestId)) {
              void game.speakerOf(feed.current.channelOf(requestId) ?? 'gm').then((name) => {
                feed.current.setSpeaker(requestId, name);
                setLive(feed.current.bubbles());
              });
            }
          }
        });
        const first = await snapshot(game);
        setState(first);
        setReady(true);
        // Nothing to decide: begin the playthrough, or open the story after the choice.
        if (first.step === 'start' || first.step === 'open') run((g, signal) => g.recover(signal));
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    return () => {
      closed = true;
      off?.();
      abortRef.current?.abort();
      liveFeed.end();
      const game = gameRef.current;
      gameRef.current = undefined;
      void Promise.resolve(runningRef.current).finally(() => game?.close());
    };
  }, [saveId, run]);

  return {
    ready,
    phase: state?.phase,
    lines: state?.lines ?? [],
    live,
    pending,
    proposal: state?.proposal,
    busy,
    thinking,
    error,
    stalled: !!state?.step,
    online,
    costs,
    sessionCost,
    send: (text) => run((g, signal) => g.send(text, signal), text),
    clarify: (text) => run((g, signal) => g.clarify(text, signal), text),
    choose: (index) => run((g, signal) => g.choose(index, signal)),
    recover: () => run((g, signal) => g.recover(signal)),
  };
}
