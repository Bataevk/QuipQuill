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

  /** Appends after the chat's last message, all or nothing. */
  async append(chatId: string, messages: BaseMessage[]): Promise<void> {
    if (messages.length === 0) return;
    const stored = mapChatMessagesToStoredMessages(messages);
    await this.db.transaction(async (tx) => {
      const row = await tx.get<{ seq: number }>(
        'SELECT coalesce(max(seq), 0) AS seq FROM model_messages WHERE chat_id = ?',
        [chatId],
      );
      let seq = Number(row?.seq ?? 0);
      for (const message of stored) {
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
    });
  }
}
