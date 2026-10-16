/**
 * One open playthrough for the UI (stage 5.2): the save, its world, the master with its
 * middleware and NPCs over the cloud queue, and the steps of a game — start, character
 * choice, turns, recovery. Screens stay thin: they draw `lines()` and the live stream from
 * `onStream()`, and call these methods.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';

import type { CharacterProposal } from '../agent/character';
import { lastProposal } from '../agent/character';
import { ChatStore, GM_CHAT } from '../agent/chat-store';
import { createGmAgent, type GmAgent } from '../agent/gm-agent';
import { queueModels } from '../agent/models';
import { isUnfinished, retryTurn, runTurn, type TurnResult } from '../agent/run-turn';
import {
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
import { QueueChatModel } from '../net/queue-chat-model';
import type { ServerEvent } from '../protocol/messages';
import { type ActionUsage, addUsage, emptyUsage, UsageMeter } from '../protocol/usage-meter';
import type { Db } from '../storage/db';
import { readStoryMeta, type StoryMeta } from '../story/story-meta';
import { nowIso } from '../util/ids';
import { openVectorIndex } from '../vector/open-index';
import type { Entity } from '../world/model';
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
  summaryModel?: BaseChatModel;
  characterModel: BaseChatModel;
  npcModel: (npc: Entity) => BaseChatModel;
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
        new QueueChatModel({ socket: init.services.socket, sessionId: init.save.id, channel: 'gm', purpose: 'gm' }),
      );
    const embeddings = init.embeddings;
    this.agent = createGmAgent({
      model: models.model,
      guardModel: models.guardModel,
      summaryModel: models.summaryModel,
      world: init.world,
      meta: init.meta,
      embed: embeddings ? async (text) => (await embeddings.embed([text]))[0] : undefined,
      npc: { model: models.npcModel, chats: this.chats, transcript: this.transcript },
    });
    this.sessionDeps = {
      world: init.world,
      chats: this.chats,
      transcript: this.transcript,
      meta: init.meta,
      characterModel: models.characterModel,
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
      return new Game({ save, db, meta, world, character, services, embeddings });
    } catch (error) {
      await db.close();
      throw error;
    }
  }

  async close(): Promise<void> {
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

  /** The pending character proposal while adapting. */
  proposal(): Promise<CharacterProposal | undefined> {
    return lastProposal(this.chats);
  }

  /** The last turn failed or was cut off and can be finished with `recover()`. */
  async unfinished(): Promise<boolean> {
    return isUnfinished(await this.chats.load(GM_CHAT));
  }

  /** Starts the playthrough with the save's character: the first proposal. */
  start(signal?: AbortSignal): Promise<CharacterProposal> {
    return this.metered(() =>
      beginSession({ ...this.sessionDeps, signal }, { storyId: this.save.storyId, character: this.sheet() }),
    );
  }

  clarify(wishes: string, signal?: AbortSignal): Promise<CharacterProposal> {
    return this.metered(() => clarifyCharacter({ ...this.sessionDeps, signal }, wishes));
  }

  choose(index: number, signal?: AbortSignal): Promise<TurnResult> {
    return this.metered(() => this.touched(chooseCharacter({ ...this.sessionDeps, agent: this.agent, signal }, index)));
  }

  send(input: string, signal?: AbortSignal): Promise<TurnResult> {
    return this.metered(() => this.touched(runTurn({ ...this.turnDeps(signal), input })));
  }

  /**
   * Picks up wherever the game stopped after a failure: starts the session or the proposal,
   * opens the story after the choice, or finishes the last turn. Undefined: nothing to do.
   */
  async recover(signal?: AbortSignal): Promise<TurnResult | CharacterProposal | undefined> {
    switch (await this.phase()) {
      case 'new':
      case 'adapting':
        return this.start(signal); // idempotent: returns the pending proposal
      case 'opening':
        return this.metered(() => this.touched(startStory({ ...this.sessionDeps, agent: this.agent, signal })));
      case 'playing':
        return (await this.unfinished())
          ? this.metered(() => this.touched(retryTurn(this.turnDeps(signal))))
          : undefined;
    }
  }

  /** Live events of this playthrough's generations (deltas, replays, statuses…). */
  onStream(listener: (event: ServerEvent) => void): () => void {
    return this.services.socket.onEvent((event) => {
      if ('sessionId' in event && event.sessionId === this.save.id) listener(event);
    });
  }

  /** The name to show for a channel: 'gm' → undefined (the narrator), 'npc:<id>' → its name. */
  async speakerOf(channel: string): Promise<string | undefined> {
    if (!channel.startsWith('npc:')) return undefined;
    const cached = this.names.get(channel);
    if (cached) return cached;
    const id = channel.slice('npc:'.length);
    const entity = (await this.world.state.entity(id)) ?? (await this.world.state.entity(channel));
    const name = entity?.name ?? id;
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
  private async metered<T>(work: () => Promise<T>): Promise<T> {
    const meter = new UsageMeter();
    const off = this.onStream((event) => meter.apply(event));
    try {
      return await work();
    } finally {
      off();
      this.lastUsage = meter.total;
      this.sessionUsage = addUsage(this.sessionUsage, meter.total);
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
