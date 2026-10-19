/**
 * Prompts live in the cloud (ADR 57): a system message names a template and gives its variables
 * as a `PromptPart`; the cloud fills it in before the model call
 * (cloud/src/quipquill_cloud/llm/prompts.py). The app never ships the prompts' text, and the same
 * variables give the same text, so the provider's cache holds as before.
 */
import { type ContentBlock, SystemMessage } from '@langchain/core/messages';

import type { PromptName, PromptPart } from '../protocol/messages';

export function promptPart(
  name: PromptName,
  vars: Record<string, string> = {},
  cache?: PromptPart['cache_control'],
): PromptPart {
  return { type: 'prompt', name, vars, ...(cache ? { cache_control: cache } : {}) };
}

/** A system message that is one template, optionally with more parts after it. */
export function promptMessage(part: PromptPart, ...rest: { type: 'text'; text: string }[]): SystemMessage {
  // LangChain does not know the part; it travels as is (net/wire-format.ts).
  return new SystemMessage({ content: [part as unknown as ContentBlock, ...rest] });
}
