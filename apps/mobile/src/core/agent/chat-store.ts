import {
  type BaseMessage,
  mapChatMessagesToStoredMessages,
  mapStoredMessagesToChatMessages,
  type StoredMessage,
} from '@langchain/core/messages';

import type { Db } from '../storage/db';

/** The master's chat; NPC chats are 'npc:<id>' (stage 4.3). */
export const GM_CHAT = 'gm';

const ROLES: Record<string, string> = { system: 'system', human: 'user', ai: 'assistant', tool: 'tool' };

/**
 * The models' working memory (`model_messages`, data-model.md): one row per LangChain message,
 * stored as its JSON form. Not what the player saw — that is `transcript`.
 */
export class ChatStore {
  private readonly db: Db;

  constructor(db: Db) {
    this.db = db;
  }

  async load(chatId: string): Promise<BaseMessage[]> {
    const rows = await this.db.all<{ content: string }>(
      'SELECT content FROM model_messages WHERE chat_id = ? ORDER BY seq',
      [chatId],
    );
    return mapStoredMessagesToChatMessages(rows.map((row) => JSON.parse(row.content) as StoredMessage));
  }

  /** Deletes the whole chat (the character adaptation chat after the choice). */
  async clear(chatId: string): Promise<void> {
    await this.db.run('DELETE FROM model_messages WHERE chat_id = ?', [chatId]);
  }

  /** Appends after the chat's last message, all or nothing. */
  async append(chatId: string, messages: BaseMessage[]): Promise<void> {
    await this.db.transaction(async (tx) => {
      const row = await tx.get<{ n: number }>('SELECT count(*) AS n FROM model_messages WHERE chat_id = ?', [
        chatId,
      ]);
      await insert(tx, chatId, Number(row?.n ?? 0), mapChatMessagesToStoredMessages(messages));
    });
  }

  /**
   * Makes the chat equal `messages` from position `from` on (0-based): later rows are replaced.
   * Rows are numbered 1..n, so a chat rewritten by summarizing stays dense.
   */
  async writeFrom(chatId: string, from: number, messages: StoredMessage[]): Promise<void> {
    await this.db.transaction(async (tx) => {
      await tx.run('DELETE FROM model_messages WHERE chat_id = ? AND seq > ?', [chatId, from]);
      await insert(tx, chatId, from, messages);
    });
  }
}

async function insert(tx: Db, chatId: string, after: number, messages: StoredMessage[]): Promise<void> {
  let seq = after;
  for (const message of messages) {
    const role = ROLES[message.type];
    if (!role) throw new Error(`Cannot store a ${message.type} message`);
    seq += 1;
    await tx.run('INSERT INTO model_messages (chat_id, seq, role, content) VALUES (?, ?, ?, ?)', [
      chatId,
      seq,
      role,
      JSON.stringify(message),
    ]);
  }
}

/**
 * Saves a chat while the agent runs: after each step only what changed is written — new
 * messages appended, or the tail rewritten when a middleware replaced or removed messages
 * (BugGuard, InvalidToolCalls, Context). A turn cut short keeps everything up to its last step.
 */
export class ChatSync {
  private readonly store: ChatStore;
  private readonly chatId: string;
  private saved: string[];

  constructor(store: ChatStore, chatId: string, saved: BaseMessage[]) {
    this.store = store;
    this.chatId = chatId;
    this.saved = mapChatMessagesToStoredMessages(saved).map((m) => JSON.stringify(m));
  }

  async save(messages: BaseMessage[]): Promise<void> {
    const stored = mapChatMessagesToStoredMessages(messages);
    const json = stored.map((m) => JSON.stringify(m));
    let same = 0;
    while (same < json.length && same < this.saved.length && json[same] === this.saved[same]) same += 1;
    if (same === json.length && same === this.saved.length) return;
    await this.store.writeFrom(this.chatId, same, stored.slice(same));
    this.saved = json;
  }
}
