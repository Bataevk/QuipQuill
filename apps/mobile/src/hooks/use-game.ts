import { useCallback, useEffect, useRef, useState } from 'react';

import type { CharacterProposal } from '@/core/agent/character';
import type { TranscriptLine } from '@/core/agent/transcript';
import { Game, type GamePhase, isStoryChannel } from '@/core/game/game';
import { StreamText } from '@/core/protocol/stream-text';
import { openGame } from '@/platform/services';

/** A bubble being streamed right now (stage 5.2): deltas by request, replaced on replay. */
export interface LiveBubble {
  requestId: string;
  channel: string;
  speaker: string | undefined;
  text: string;
}

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
  unfinished: boolean;
  send(text: string): void;
  clarify(text: string): void;
  choose(index: number): void;
  recover(): void;
}

/**
 * The game screen's state over one `Game`: committed lines from the transcript, live bubbles
 * from the socket stream while a turn runs, one action at a time. Leaving the screen aborts a
 * running action and closes the save when it settles.
 */
export function useGame(saveId: string): GameView {
  const gameRef = useRef<Game | undefined>(undefined);
  const streams = useRef(new Map<string, { stream: StreamText; channel: string; speaker?: string }>());
  const abortRef = useRef<AbortController | undefined>(undefined);
  const runningRef = useRef<Promise<unknown> | undefined>(undefined);
  const [ready, setReady] = useState(false);
  const [phase, setPhase] = useState<GamePhase>();
  const [lines, setLines] = useState<TranscriptLine[]>([]);
  const [live, setLive] = useState<LiveBubble[]>([]);
  const [pending, setPending] = useState<string>();
  const [proposal, setProposal] = useState<CharacterProposal>();
  const [busy, setBusy] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string>();
  const [unfinished, setUnfinished] = useState(false);

  const publishLive = useCallback(() => {
    setLive(
      [...streams.current.entries()]
        .map(([requestId, s]) => ({ requestId, channel: s.channel, speaker: s.speaker, text: s.stream.text }))
        .filter((bubble) => bubble.text.trim()),
    );
  }, []);

  const refresh = useCallback(async () => {
    const game = gameRef.current;
    if (!game) return;
    const nextPhase = await game.phase();
    const [nextLines, nextProposal, nextUnfinished] = await Promise.all([
      game.lines(),
      nextPhase === 'adapting' ? game.proposal() : Promise.resolve(undefined),
      nextPhase === 'playing' ? game.unfinished() : Promise.resolve(false),
    ]);
    setPhase(nextPhase);
    setLines(nextLines);
    setProposal(nextProposal);
    setUnfinished(nextUnfinished);
  }, []);

  const run = useCallback(
    (action: (game: Game, signal: AbortSignal) => Promise<unknown>, shown?: string) => {
      const game = gameRef.current;
      if (!game || runningRef.current) return;
      const controller = new AbortController();
      abortRef.current = controller;
      setBusy(true);
      setError(undefined);
      setPending(shown);
      streams.current.clear();
      publishLive();
      const work = action(game, controller.signal)
        .catch((e: unknown) => {
          if (!controller.signal.aborted) setError(e instanceof Error ? e.message : String(e));
        })
        .then(() => (controller.signal.aborted ? undefined : refresh()))
        .finally(() => {
          runningRef.current = undefined;
          if (controller.signal.aborted) return;
          streams.current.clear();
          publishLive();
          setPending(undefined);
          setThinking(false);
          setBusy(false);
        });
      runningRef.current = work;
    },
    [publishLive, refresh],
  );

  useEffect(() => {
    let closed = false;
    let off: (() => void) | undefined;
    openGame(saveId)
      .then(async (game) => {
        if (closed) return game.close();
        gameRef.current = game;
        off = game.onStream((event) => {
          if (event.type === 'status') return setThinking(true);
          if (!('requestId' in event) || !event.requestId || !isStoryChannel(event.channel)) return;
          let entry = streams.current.get(event.requestId);
          if (!entry) {
            entry = { stream: new StreamText(event.requestId), channel: event.channel ?? 'gm' };
            streams.current.set(event.requestId, entry);
            const speaker = event.meta?.speaker;
            if (typeof speaker === 'string') entry.speaker = speaker;
            else void game.speakerOf(entry.channel).then((name) => ((entry!.speaker = name), publishLive()));
          }
          if (entry.stream.apply(event)) {
            setThinking(false);
            publishLive();
          }
        });
        await refresh();
        setReady(true);
        if ((await game.phase()) === 'new') run((g, signal) => g.start(signal));
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    return () => {
      closed = true;
      off?.();
      abortRef.current?.abort();
      const game = gameRef.current;
      gameRef.current = undefined;
      void Promise.resolve(runningRef.current).finally(() => game?.close());
    };
  }, [saveId, refresh, run, publishLive]);

  return {
    ready,
    phase,
    lines,
    live,
    pending,
    proposal,
    busy,
    thinking,
    error,
    unfinished,
    send: (text) => run((g, signal) => g.send(text, signal), text),
    clarify: (text) => run((g, signal) => g.clarify(text, signal), text),
    choose: (index) => run((g, signal) => g.choose(index, signal)),
    recover: () => run((g, signal) => g.recover(signal)),
  };
}
