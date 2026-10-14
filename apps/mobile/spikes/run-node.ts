// Runs the pure-TS part of the 3.1 spikes in Node: `node spikes/run-node.ts`.
// @ts-expect-error -- Node's type stripping needs the explicit .ts extension.
import { runAgentSpike } from './agent.ts';

const result = await runAgentSpike();
console.log(JSON.stringify(result, null, 2));
process.exit(result.ok ? 0 : 1);
