import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import {
  BaseChatModel,
  type BaseChatModelCallOptions,
  type BindToolsInput,
} from '@langchain/core/language_models/chat_models';
import type { BaseMessage } from '@langchain/core/messages';
import type { ChatResult } from '@langchain/core/outputs';
import { convertToOpenAITool } from '@langchain/core/utils/function_calling';

import type { LlmRequest, ModelClass, Purpose, WireTool } from '../protocol/messages';
import { newId } from '../util/ids';
import type { CloudSocket } from './cloud-socket';
import { fromDone, toWireMessages } from './wire-format';

export interface QueueChatModelFields {
  socket: CloudSocket;
  /** The save (playthrough) id — the server groups resume and replay by it. */
  sessionId: string;
  /** Where the UI draws this stream: 'gm', 'npc:<id>', 'guard'… */
  channel: string;
  /** The server picks model, effort and max_tokens by purpose (ADR 10). */
  purpose: Purpose;
  /** The model class; the server picks the model and providers for it (ADR 50). Default 'basic'. */
  model?: ModelClass;
  meta?: Record<string, unknown>;
  tools?: WireTool[];
  responseFormat?: Record<string, unknown>;
}

/**
 * A LangChain chat model whose "provider" is the cloud queue (ADR 25): one `llm.request` per
 * call, deltas go to the UI through the socket's event stream (and LangChain token callbacks),
 * and the result is `done.message` — the source of truth, with complete tool calls.
 */
export class QueueChatModel extends BaseChatModel {
  readonly fields: QueueChatModelFields;

  constructor(fields: QueueChatModelFields) {
    super({});
    this.fields = fields;
  }

  _llmType(): string {
    return 'quipquill-queue';
  }

  /** Same model, another channel/purpose (e.g. an NPC chat or the guard). */
  with(fields: Partial<QueueChatModelFields>): QueueChatModel {
    return new QueueChatModel({ ...this.fields, ...fields });
  }

  bindTools(tools: BindToolsInput[]): QueueChatModel {
    return this.with({ tools: tools.map((tool) => convertToOpenAITool(tool) as WireTool) });
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'] & BaseChatModelCallOptions,
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    const { socket, sessionId, channel, purpose, model = 'basic', meta, tools, responseFormat } = this.fields;
    const request: LlmRequest = {
      v: 1,
      type: 'llm.request',
      requestId: newId(),
      sessionId,
      channel,
      purpose,
      model,
      messages: toWireMessages(messages),
      ...(tools?.length ? { tools } : {}),
      ...(responseFormat ? { responseFormat } : {}),
      ...(meta ? { meta } : {}),
    };
    const unsubscribe = socket.onEvent((event) => {
      if (event.type === 'delta' && event.requestId === request.requestId) {
        void runManager?.handleLLMNewToken(event.text);
      }
    });
    try {
      const done = await socket.request(request, options.signal);
      const message = fromDone(done);
      return { generations: [{ text: done.message.content ?? '', message }], llmOutput: { usage: done.usage } };
    } finally {
      unsubscribe();
    }
  }
}
