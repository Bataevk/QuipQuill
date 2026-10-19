/// <reference types="node" />
/**
 * The cloud's prompt templates (ADR 57) filled in here for core tests, by the same rules as the
 * cloud (cloud/src/quipquill_cloud/llm/prompts.py): every `{{VAR}}` the template uses, no others.
 * So a test sees the text the model would see, and a variable the app forgets or misnames fails.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import type { BaseMessage } from '@langchain/core/messages';

import type { PromptPart } from '@/core/protocol/messages';

const PROMPTS_DIR = fileURLToPath(new URL('../../../../../prompts', import.meta.url));
const VAR = /\{\{([A-Z][A-Z0-9_]*)\}\}/g;

export function renderPrompt({ name, vars }: PromptPart): string {
  const template = readFileSync(`${PROMPTS_DIR}/${name}.md`, 'utf8').trimEnd();
  const wanted = [...new Set([...template.matchAll(VAR)].map((m) => m[1]))].sort();
  const given = Object.keys(vars).sort();
  if (wanted.join() !== given.join()) throw new Error(`prompt ${name} takes [${wanted}], got [${given}]`);
  return template.replace(VAR, (_, key: string) => vars[key]);
}

type Part = { type?: string; text?: string; cache_control?: unknown };

/** The message's parts as the model gets them: prompt parts filled in. */
export function renderedParts(message: BaseMessage): { text: string; cache_control?: unknown }[] {
  const content = message.content;
  if (typeof content === 'string') return [{ text: content }];
  return (content as Part[]).flatMap((part) => {
    const cache = part.cache_control ? { cache_control: part.cache_control } : {};
    if (part.type === 'prompt') return [{ text: renderPrompt(part as unknown as PromptPart), ...cache }];
    return part.type === 'text' ? [{ text: part.text ?? '', ...cache }] : [];
  });
}

/** The whole text of the message as the model gets it. */
export function renderedText(message: BaseMessage): string {
  return renderedParts(message)
    .map((p) => p.text)
    .join('');
}
