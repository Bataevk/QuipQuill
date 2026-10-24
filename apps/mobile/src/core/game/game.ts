/**
 * One open playthrough for the UI (stage 5.2): the save, its world, the master with its
 * middleware and NPCs over the cloud queue, and the steps of a game — start, character
 * choice, turns, recovery. Screens stay thin: they draw `lines()` and the live stream from
 * `onStream()`, and call these methods.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';

import type { CharacterProposal } from '../agent/character';
import type { ReportContext } from '../net/auth';
import { lastProposal } from '../agent/character';
import { ChatStore, GM_CHAT } from '../agent/chat-store';
import { BackgroundWork } from '../agent/background';
import { type AfterTurnOptions, startAfterTurn, turnToSettle } from '../agent/middleware/after-turn';
import type { GeneratorResult } from '../agent/generator';
import { createGmAgent, type GmAgent } from '../agent/gm-agent';
import type { SheetGuardModel } from '../agent/sheet-check';
import { queueModels } from '../agent/models';
import type { Entity } from '../world/model';
import { isUnfinished, retryTurn, runTurn, type TurnResult } from '../agent/run-turn';
import {
  beginAuthoredSession,
  beginSession,
  chooseCharacter,
  clarifyCharacter,
  type SessionDeps,
  startStory,
} from '../agent/session';
import { TranscriptStore, type TranscriptLine } from '../agent/transcript';
import type { Character, SaveRecord } from '../library/app-db';
import type { Library } from '../library/library';
import type { CloudSocket } from '../net/cloud-socket';
import { entityIdOf, GM_CHANNEL } from '../protocol/channels';
import { QueueChatModel } from '../net/queue-chat-model';
import type { ServerEvent } from '../protocol/messages';
import { type ActionUsage, addUsage, emptyUsage, UsageMeter } from '../protocol/usage-meter';
import type { Db } from '../storage/db';
import { readStoryMeta, type StoryMeta } from '../story/story-meta';
import { nowIso } from '../util/ids';
import { openVectorIndex } from '../vector/open-index';
import type { CharacterSheet } from '../world/world';
import { World } from '../world/world';

type Embeddings = { embed(texts: string[]): Promise<Float32Array[]> };

export interface GameServices {
  library: Library;
  socket: CloudSocket;
  /**
   * Embeddings with the story's model (CloudEmbeddings): lore search queries and the vectors of
   * new and rewritten entities; without it no lore search.
   */
  embeddings?: (model: string) => Embeddings;
  /** The models; default: `queueModels` over the socket (tests pass scripted ones). */
  models?: (sessionId: string) => GameModels;
}

export interface GameModels {
  model: BaseChatModel;
  guardModel?: BaseChatModel;
  /** The Guard on the character sheet and wishes (stage T.7). */
  sheetGuardModel?: SheetGuardModel;
  /** The answer to a rejected input (stage T.6). */
  miniGmModel?: BaseChatModel;
  /** Before Generator: drafts and claims before the master (stage T.8). */
  pregenModel?: BaseChatModel;
  summaryModel?: BaseChatModel;
  characterModel: BaseChatModel;
  /** The actor pack: an entity's line on its own channel (react_entity). */
  actorModel: (entity: Entity) => BaseChatModel;
  /** The background world generator after each player turn; without it there is none. */
  generatorModel?: BaseChatModel;
  /** Entities' notes from their overflowing chats, after the generator (ADR 45); none without it. */
  memoryModel?: BaseChatModel;
}

/**
 * - `new`: the playthrough has not started (start it with `start()`);
 * - `adapting`: the character is being fitted to the world (cards: `proposal()`);
 * - `opening`: the character is chosen, the master's first turn has not begun (`recover()`);
 * - `playing`: turns; a turn that failed (the opening one too) shows as `unfinished()`.
 */
export type GamePhase = 'new' | 'adapting' | 'opening' | 'playing';

/**
 * What the game needs before the player can act (after a failure or an app restart):
 * - `start`: no proposal yet (new playthrough, or the first proposal failed) — safe to run;
 * - `open`: the character is chosen but the story has not begun — safe to run;
 * - `retry`: the last turn did not finish — ask the player (it costs a model call they did
 *   not ask for just now).
 * All three are done by `recover()`.
 */
export type GameStep = 'start' | 'open' | 'retry' | undefined;

export { isStoryChannel } from '../protocol/live-feed';

export class Game {
  readonly save: SaveRecord;
  readonly meta: StoryMeta;
  readonly world: World;
  readonly character: Character | undefined;
  private readonly db: Db;
  private readonly services: GameServices;
  private readonly chats: ChatStore;
  private readonly transcript: TranscriptStore;
  private readonly agent: GmAgent;
  private readonly sessionDeps: SessionDeps;
  private readonly names = new Map<string, string>();
  /** What the last action cost (stage 6.1); undefined before the first one. */
  lastUsage: ActionUsage | undefined;
  /** What this game cost since it was opened. */
  sessionUsage: ActionUsage = emptyUsage();
  /** Actions metered so far: which one `lastUsage` belongs to. */
  actions = 0;
  /** What the generator did after the last player turn (dev diagnostics). */
  lastGeneration: GeneratorResult | { error: string } | undefined;
  /** The generator and entity memory after a player's turn (AfterTurn middleware, stage 6.2i). */
  private readonly background: BackgroundWork;
  private readonly afterTurn: AfterTurnOptions;
  /** Resolves when every action started so far has its metering done; background work adds its cost after. */
  private acting: Promise<void> = Promise.resolve();
  private readonly generatedListeners = new Set<() => void>();

  private constructor(init: {
    save: SaveRecord;
    db: Db;
    meta: StoryMeta;
    world: World;
    character: Character | undefined;
    services: GameServices;
    embeddings: Embeddings | undefined;
  }) {
    this.save = init.save;
    this.db = init.db;
    this.meta = init.meta;
    this.world = init.world;
    this.character = init.character;
    this.services = init.services;
    this.chats = new ChatStore(init.db);
    this.transcript = new TranscriptStore(init.db);
    const models: GameModels =
      init.services.models?.(init.save.id) ??
      queueModels(
        new QueueChatModel({ socket: init.services.socket, sessionId: init.save.id, channel: GM_CHANNEL, purpose: 'gm' }),
      );
    const embeddings = init.embeddings;
    this.background = new BackgroundWork({ around: (work) => this.meteredBackground(work) });
    this.afterTurn = {
      world: init.world,
      meta: init.meta,
      chats: this.chats,
      work: this.background,
      generatorModel: models.generatorModel,
      memoryModel: models.memoryModel,
      onGenerated: (result) => (this.lastGeneration = result),
    };
    this.agent = createGmAgent({
      model: models.model,
      guardModel: models.guardModel,
      miniGmModel: models.miniGmModel,
      pregenModel: models.pregenModel,
      summaryModel: models.summaryModel,
      world: init.world,
      meta: init.meta,
      embed: embeddings ? async (text) => (await embeddings.embed([text]))[0] : undefined,
      actor: { model: models.actorModel, chats: this.chats },
      afterTurn: this.afterTurn,
    });
    this.sessionDeps = {
      world: init.world,
      chats: this.chats,
      transcript: this.transcript,
      meta: init.meta,
      characterModel: models.characterModel,
      sheetGuard: models.sheetGuardModel,
    };
  }

  static async open(services: GameServices, saveId: string): Promise<Game> {
    const { repository, saves } = services.library;
    const save = await repository.getSave(saveId);
    if (!save) throw new Error(`There is no save ${saveId}`);
    const db = await saves.open(saveId);
    try {
      const meta = await readStoryMeta(db);
      const embeddings = services.embeddings?.(meta.embedModel);
      const world = await World.open(db, {
        vectors: await openVectorIndex(db),
        embed: embeddings && ((texts) => embeddings.embed(texts)),
      });
      // Vectors that could not be computed last time (no network): catch up in the background.
      void world.refreshLore().catch(() => 0);
      const characters = await repository.listCharacters();
      const character = characters.find((c) => c.id === save.characterId);
      const game = new Game({ save, db, meta, world, character, services, embeddings });
      await game.finishLostAfterTurn();
      return game;
    } catch (error) {
      await db.close();
      throw error;
    }
  }

  async close(): Promise<void> {
    this.background.cancel();
    await this.settle();
    await this.db.close();
  }

  async phase(): Promise<GamePhase> {
    const { log } = this.world;
    if ((await log.count('session.started')) === 0) return 'new';
    if ((await log.count('character.adapted')) === 0) return 'adapting';
    return (await this.chats.load(GM_CHAT)).length === 0 ? 'opening' : 'playing';
  }

  async nextStep(): Promise<GameStep> {
    switch (await this.phase()) {
      case 'new':
        return 'start';
      case 'adapting':
        return (await this.proposal()) ? undefined : 'start';
      case 'opening':
        return 'open';
      case 'playing':
        return (await this.unfinished()) ? 'retry' : undefined;
    }
  }

  /** Everything the player has seen, oldest first. */
  lines(): Promise<TranscriptLine[]> {
    return this.transcript.lines();
  }

  /**
   * What the Report button sends besides the player's words: the story and save, the hero as
   * created and as the world has it now, and every line the player has seen.
   */
  async reportContext(): Promise<ReportContext> {
    const [lines, player] = await Promise.all([this.transcript.lines(), this.world.state.player()]);
    return {
      storyId: this.save.storyId,
      saveId: this.save.id,
      character: { sheet: this.sheet(), inWorld: player ?? null },
      transcript: lines.map(({ seq, speaker, text, ts }) => ({ seq, speaker, text, ts })),
    };
  }

  /** The pending character proposal while adapting. */
  proposal(): Promise<CharacterProposal | undefined> {
    return lastProposal(this.chats);
  }

  /** The last turn failed or was cut off and can be finished with `recover()`. */
  async unfinished(): Promise<boolean> {
    return isUnfinished(await this.chats.load(GM_CHAT));
  }

  /** A hero of the story itself (ADR 62): played as written, without adaptation. */
  get authored(): boolean {
    return !!this.character?.storyId;
  }

  /**
   * Starts the playthrough with the save's character: the first proposal — or, for a hero of the
   * story itself, straight to the opening scene (ADR 62).
   */
  async start(signal?: AbortSignal): Promise<CharacterProposal | TurnResult> {
    await this.settle();
    const input = { storyId: this.save.storyId, character: this.sheet() };
    if (this.authored) {
      return this.metered(() => this.touched(beginAuthoredSession({ ...this.sessionDeps, agent: this.agent, signal }, input)));
    }
    return this.metered(() => beginSession({ ...this.sessionDeps, signal }, input));
  }

  async clarify(wishes: string, signal?: AbortSignal): Promise<CharacterProposal> {
    await this.settle();
    return this.metered(() => clarifyCharacter({ ...this.sessionDeps, signal }, wishes));
  }

  /** A card of proposal `round` (the latest by default; earlier cards stay choosable, stage T.3). */
  async choose(index: number, round?: number, signal?: AbortSignal): Promise<TurnResult> {
    await this.settle();
    return this.metered(() =>
      this.touched(chooseCharacter({ ...this.sessionDeps, agent: this.agent, signal }, index, round)),
    );
  }

  /**
   * A player turn. When it ends, the generator brings the world in line with it in the
   * background (`whenGenerated`); the next action waits for it first.
   */
  async send(input: string, signal?: AbortSignal): Promise<TurnResult> {
    await this.settle();
    return this.metered(() => this.touched(runTurn({ ...this.turnDeps(signal), input })));
  }

  /**
   * Picks up wherever the game stopped after a failure: starts the session or the proposal,
   * opens the story after the choice, or finishes the last turn. Undefined: nothing to do.
   */
  async recover(signal?: AbortSignal): Promise<TurnResult | CharacterProposal | undefined> {
    await this.settle();
    switch (await this.phase()) {
      case 'new':
      case 'adapting':
        return this.start(signal); // idempotent: returns the pending proposal
      case 'opening':
        return this.metered(() => this.touched(startStory({ ...this.sessionDeps, agent: this.agent, signal })));
      case 'playing': {
        if (!(await this.unfinished())) return undefined;
        return this.metered(() => this.touched(retryTurn(this.turnDeps(signal))));
      }
    }
  }

  /** Resolves when the generator of the last turn (if any) has finished, failed or timed out. */
  whenGenerated(): Promise<void> {
    return this.settle();
  }

  /** Called when the generator finishes: `lastUsage` / `sessionUsage` now include its cost. */
  onGenerated(listener: () => void): () => void {
    this.generatedListeners.add(listener);
    return () => this.generatedListeners.delete(listener);
  }

  /** Live events of this playthrough's generations (deltas, replays, statuses…). */
  onStream(listener: (event: ServerEvent) => void): () => void {
    return this.services.socket.onEvent((event) => {
      if ('sessionId' in event && event.sessionId === this.save.id) listener(event);
    });
  }

  /** The name to show for a channel: 'gm' → undefined (the narrator), 'npc:<id>' → its name. */
  async speakerOf(channel: string): Promise<string | undefined> {
    const id = entityIdOf(channel);
    if (!id) return undefined;
    const cached = this.names.get(channel);
    if (cached) return cached;
    const name = (await this.world.state.entity(id))?.name ?? id.replace(/^npc:/, '');
    this.names.set(channel, name);
    return name;
  }

  private sheet(): CharacterSheet {
    const c = this.character;
    if (!c) return { id: null, name: 'Stranger', description: '' };
    return { id: c.id, name: c.name, sex: c.sex, age: c.age, race: c.race, description: c.description };
  }

  private turnDeps(signal?: AbortSignal) {
    return { agent: this.agent, world: this.world, chats: this.chats, transcript: this.transcript, signal };
  }

  /** Counts what the generations of one action cost (`lastUsage`, `sessionUsage`). */
  private metered<T>(work: () => Promise<T>): Promise<T> {
    let release!: () => void;
    const finished = new Promise<void>((resolve) => (release = resolve));
    // Chained, not replaced: background work waits for every action in flight, not only the last.
    this.acting = Promise.all([this.acting, finished]).then(() => undefined);
    return this.measure(work, (usage) => {
      this.actions += 1;
      this.lastUsage = usage;
      this.sessionUsage = addUsage(this.sessionUsage, usage);
      release();
    });
  }

  /**
   * The app was killed after a turn, before its background work finished (ADR 48): run it now,
   * in the background as usual. A turn still unfinished is left to `recover` (its retry runs it).
   */
  private async finishLostAfterTurn(): Promise<void> {
    const history = await this.chats.load(GM_CHAT);
    if (isUnfinished(history)) return;
    const turn = turnToSettle(history);
    if (turn && !(await this.world.afterTurnDone(turn.turn))) startAfterTurn(this.afterTurn, turn);
  }

  /** Waits for the background work of the last turn; it ends by itself within AFTER_TURN_TIMEOUT_MS. */
  private settle(): Promise<void> {
    return this.background.settle();
  }

  /**
   * Meters one background job: its cost joins the turn's (`lastUsage`) once the turn's own
   * metering is done, then `onGenerated` listeners hear of it.
   */
  private meteredBackground(work: () => Promise<void>): Promise<void> {
    return this.measure(work, async (usage) => {
      await this.acting;
      this.lastUsage = addUsage(this.lastUsage ?? emptyUsage(), usage);
      this.sessionUsage = addUsage(this.sessionUsage, usage);
      this.generatedListeners.forEach((listener) => listener());
    });
  }

  /** Runs `work` while counting this playthrough's generations; `done` gets their cost, failed or not. */
  private async measure<T>(work: () => Promise<T>, done: (usage: ActionUsage) => void | Promise<void>): Promise<T> {
    const meter = new UsageMeter();
    const off = this.onStream((event) => meter.apply(event));
    try {
      return await work();
    } finally {
      off();
      await done(meter.total);
    }
  }

  private async touched<T>(work: Promise<T>): Promise<T> {
    try {
      return await work;
    } finally {
      await this.services.library.repository.touchSave(this.save.id, nowIso()).catch(() => undefined);
    }
  }
}
