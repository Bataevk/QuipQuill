import { HttpError, requestJson } from './http';

/** What the cloud returns on login, register and refresh (protocol.md#авторизация-rest). */
interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface Tokens {
  accessToken: string;
  refreshToken: string;
  /** Epoch ms when the access token expires. */
  accessExpiresAt: number;
}

/**
 * Where tokens survive app restarts. Port: expo-secure-store in the app
 * (src/platform/secure-token-store.ts), memory in tests.
 */
export interface TokenStore {
  load(): Promise<Tokens | null>;
  save(tokens: Tokens): Promise<void>;
  clear(): Promise<void>;
}

export class MemoryTokenStore implements TokenStore {
  private tokens: Tokens | null = null;

  async load(): Promise<Tokens | null> {
    return this.tokens;
  }

  async save(tokens: Tokens): Promise<void> {
    this.tokens = tokens;
  }

  async clear(): Promise<void> {
    this.tokens = null;
  }
}

/** The player must log in again (no tokens, or the refresh token was rejected). */
/** The signed-in account as the profile shows it; `email` is empty for a guest (ADR 60). */
export interface Account {
  email: string | null;
  tier: string;
  kind: 'guest' | 'email';
  termsVersion: string | null;
  /** A guest's allowance for the whole prototype, USD (ADR 66); null for email accounts. */
  totalLimitUsd: number | null;
  /** Spent so far, all days together, USD. */
  spentUsd: number;
}

/** Where the player left an email for the release waitlist (ADR 66). */
export type WaitlistSource = 'limit' | 'profile' | 'onboarding';

/** The game as a report shows it (`Game.reportContext()`). */
export interface ReportContext {
  storyId: string;
  saveId: string;
  character: Record<string, unknown>;
  transcript: { seq: number; speaker: string; text: string; ts: string }[];
}

/** A player's report (`POST /reports`). */
export interface GameReport {
  message: string;
  context: ReportContext;
  /** The app's platform and version, for reproducing. */
  client: Record<string, unknown>;
}

export class AuthRequiredError extends Error {
  constructor(message = 'Sign in required') {
    super(message);
    this.name = 'AuthRequiredError';
  }
}

/** Refresh this long before the access token expires (clock skew, slow networks). */
const REFRESH_MARGIN_MS = 60_000;

/**
 * The signed-in session: login / register, and a valid access token on demand. The refresh
 * token rotates on every use (ADR 17), so concurrent callers share one refresh in flight.
 */
export class AuthSession {
  private readonly baseUrl: string;
  private readonly store: TokenStore;
  private readonly now: () => number;
  private refreshing: Promise<Tokens> | undefined;

  constructor(baseUrl: string, store: TokenStore, now: () => number = Date.now) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.store = store;
    this.now = now;
  }

  async isSignedIn(): Promise<boolean> {
    return (await this.store.load()) !== null;
  }

  async login(email: string, password: string): Promise<void> {
    await this.keep(await requestJson<TokenPair>(`${this.baseUrl}/auth/login`, { body: { email, password } }));
  }

  async register(email: string, password: string, inviteCode: string): Promise<void> {
    await this.keep(
      await requestJson<TokenPair>(`${this.baseUrl}/auth/register`, {
        body: { email, password, invite_code: inviteCode },
      }),
    );
  }

  /** A new guest account for this device (ADR 60): no email, no password. */
  async guest(): Promise<void> {
    await this.keep(await requestJson<TokenPair>(`${this.baseUrl}/auth/guest`, { method: 'POST' }));
  }

  /** The player accepted these terms (ADR 60); the cloud keeps the version and the time. */
  async consent(termsVersion: string): Promise<void> {
    await requestJson(`${this.baseUrl}/me/consent`, {
      body: { terms_version: termsVersion },
      token: await this.accessToken(),
    });
  }

  /** Who is signed in (`GET /me`): the profile shows it. */
  async me(): Promise<Account> {
    const me = await requestJson<{
      email: string | null;
      tier: string;
      kind?: string;
      terms_version?: string | null;
      limits?: { total_cost_usd?: string | null; spent_total_usd?: string };
    }>(`${this.baseUrl}/me`, { token: await this.accessToken() });
    const total = me.limits?.total_cost_usd;
    return {
      email: me.email,
      tier: me.tier,
      kind: me.kind === 'guest' ? 'guest' : 'email',
      termsVersion: me.terms_version ?? null,
      totalLimitUsd: total === null || total === undefined ? null : Number(total),
      spentUsd: Number(me.limits?.spent_total_usd ?? 0),
    };
  }

  /** Leaves an email for the release waitlist (ADR 66); the same email again changes nothing. */
  async joinWaitlist(email: string, source: WaitlistSource): Promise<void> {
    await requestJson(`${this.baseUrl}/waitlist`, { body: { email, source }, token: await this.accessToken() });
  }

  /** The game's Report button: the player's words, the game as it was and the app's version. */
  async sendReport(report: GameReport): Promise<void> {
    const { message, context, client } = report;
    await requestJson(`${this.baseUrl}/reports`, {
      body: {
        message,
        story_id: context.storyId,
        save_id: context.saveId,
        character: context.character,
        transcript: context.transcript,
        client,
      },
      token: await this.accessToken(),
      timeoutMs: 30_000,
    });
  }

  async logout(): Promise<void> {
    await this.store.clear();
  }

  /** A valid access token, refreshed if it is about to expire (or `force`). */
  async accessToken(options: { force?: boolean } = {}): Promise<string> {
    const tokens = await this.store.load();
    if (!tokens) throw new AuthRequiredError();
    if (!options.force && tokens.accessExpiresAt - REFRESH_MARGIN_MS > this.now()) {
      return tokens.accessToken;
    }
    this.refreshing ??= this.refresh(tokens.refreshToken).finally(() => {
      this.refreshing = undefined;
    });
    return (await this.refreshing).accessToken;
  }

  private async refresh(refreshToken: string): Promise<Tokens> {
    try {
      return await this.keep(
        await requestJson<TokenPair>(`${this.baseUrl}/auth/refresh`, {
          body: { refresh_token: refreshToken },
        }),
      );
    } catch (error) {
      if (error instanceof HttpError && error.status === 401) {
        await this.store.clear(); // expired, revoked or reused: sign in again
        throw new AuthRequiredError(error.detail);
      }
      throw error; // network trouble: keep the tokens, try again later
    }
  }

  private async keep(pair: TokenPair): Promise<Tokens> {
    const tokens: Tokens = {
      accessToken: pair.access_token,
      refreshToken: pair.refresh_token,
      accessExpiresAt: this.now() + pair.expires_in * 1000,
    };
    await this.store.save(tokens);
    return tokens;
  }
}
