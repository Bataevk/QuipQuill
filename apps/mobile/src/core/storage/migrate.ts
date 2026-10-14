import type { Db } from './db';

/**
 * Applies schema steps not yet applied, tracked in `PRAGMA user_version`: step i (0-based)
 * moves the database from version i to i + 1. Steps are append-only; never edit a released one.
 */
export async function migrate(db: Db, steps: readonly string[]): Promise<number> {
  const row = await db.get<{ user_version: number }>('PRAGMA user_version');
  const current = Number(row?.user_version ?? 0);
  if (current > steps.length) {
    throw new Error(`Database schema v${current} is newer than this app (v${steps.length})`);
  }
  for (let version = current; version < steps.length; version++) {
    await db.transaction(async (tx) => {
      await tx.exec(steps[version]);
      await tx.exec(`PRAGMA user_version = ${version + 1}`);
    });
  }
  return steps.length;
}
