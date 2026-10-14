/** fetch with an explicit timeout and JSON in/out (CLAUDE.md: every external call has one). */

export const DEFAULT_TIMEOUT_MS = 15_000;

/** A non-2xx answer; `detail` is the server's error code (e.g. "invalid_credentials"). */
export class HttpError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(`HTTP ${status}: ${detail}`);
    this.name = 'HttpError';
    this.status = status;
    this.detail = detail;
  }
}

export async function requestJson<T>(
  url: string,
  init: { method?: string; body?: unknown; token?: string; timeoutMs?: number } = {},
): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  if (init.body !== undefined) headers['Content-Type'] = 'application/json';
  if (init.token) headers.Authorization = `Bearer ${init.token}`;
  const response = await fetch(url, {
    method: init.method ?? (init.body === undefined ? 'GET' : 'POST'),
    headers,
    body: init.body === undefined ? undefined : JSON.stringify(init.body),
    signal: AbortSignal.timeout(init.timeoutMs ?? DEFAULT_TIMEOUT_MS),
  });
  const text = await response.text();
  const data = text ? (JSON.parse(text) as unknown) : undefined;
  if (!response.ok) {
    const detail = (data as { detail?: unknown } | undefined)?.detail;
    throw new HttpError(response.status, typeof detail === 'string' ? detail : response.statusText);
  }
  return data as T;
}
