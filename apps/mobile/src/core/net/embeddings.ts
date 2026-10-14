import { AuthRequiredError, type AuthSession } from './auth';
import { HttpError, requestJson } from './http';

/** Server limits (protocol.md#эмбеддинги-rest). */
const MAX_INPUTS = 64;

/**
 * Query vectors for lore search via the cloud (`POST /embeddings`). The model must be the
 * story's `meta.embed_model`, or the vectors will not be comparable.
 */
export class CloudEmbeddings {
  private readonly baseUrl: string;
  private readonly auth: AuthSession;
  private readonly model: string;

  constructor(baseUrl: string, auth: AuthSession, model: string) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.auth = auth;
    this.model = model;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    const vectors: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += MAX_INPUTS) {
      vectors.push(...(await this.batch(texts.slice(i, i + MAX_INPUTS))));
    }
    return vectors;
  }

  private async batch(input: string[], retried = false): Promise<Float32Array[]> {
    const token = await this.auth.accessToken({ force: retried });
    try {
      const data = await requestJson<{ vectors: number[][] }>(`${this.baseUrl}/embeddings`, {
        body: { model: this.model, input },
        token,
        timeoutMs: 30_000,
      });
      return data.vectors.map((vector) => Float32Array.from(vector));
    } catch (error) {
      // An access token can expire between the check and the call: refresh once.
      if (error instanceof HttpError && error.status === 401 && !retried) return this.batch(input, true);
      if (error instanceof HttpError && error.status === 401) throw new AuthRequiredError(error.detail);
      throw error;
    }
  }
}
