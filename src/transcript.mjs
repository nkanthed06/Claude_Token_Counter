import { open, stat } from 'node:fs/promises';

/** Transcripts grow without bound; the newest entries are all we need. */
const TAIL_BYTES = 512 * 1024;

/**
 * Reads the size of the context that will be re-sent on the next turn, plus the
 * model currently answering, from the session transcript.
 *
 * The last assistant message's usage tells us exactly how many tokens were in
 * the window when it was produced. Its own output then joins the context for
 * the following turn, so it is counted too.
 *
 * Returns zeroed values for a first prompt, an unreadable path, or a transcript
 * with no assistant turn yet. The caller must never fail because of this.
 */
export async function readContextState(transcriptPath) {
  const empty = { contextTokens: 0, model: null, available: false };
  if (typeof transcriptPath !== 'string' || transcriptPath === '') return empty;

  let text;
  let truncated;
  try {
    const { size } = await stat(transcriptPath);
    const start = Math.max(0, size - TAIL_BYTES);
    truncated = start > 0;
    const handle = await open(transcriptPath, 'r');
    try {
      const buffer = Buffer.alloc(Math.min(size, TAIL_BYTES));
      await handle.read(buffer, 0, buffer.length, start);
      text = buffer.toString('utf8');
    } finally {
      await handle.close();
    }
  } catch {
    return empty;
  }

  const lines = text.split('\n');
  // Only skip the opening line when the tail read actually cut into a record.
  // On a transcript small enough to read whole, line 0 is a complete entry.
  const lowest = truncated ? 1 : 0;
  for (let index = lines.length - 1; index >= lowest; index -= 1) {
    const line = lines[index].trim();
    if (line === '') continue;

    let entry;
    try {
      entry = JSON.parse(line);
    } catch {
      continue;
    }
    if (entry?.type !== 'assistant') continue;

    const message = entry.message ?? {};
    const usage = message.usage;
    if (!usage || typeof usage !== 'object') continue;

    const contextTokens =
      count(usage.input_tokens) +
      count(usage.cache_creation_input_tokens) +
      count(usage.cache_read_input_tokens) +
      count(usage.output_tokens);

    return {
      contextTokens,
      model: typeof message.model === 'string' ? message.model : null,
      available: true,
    };
  }

  return empty;
}

/**
 * Accumulates all tokens billed across the session by reading every assistant
 * turn in the transcript. Used to show the user how much they've spent.
 *
 * Pricing is computed using the published rates for each turn's model. If a
 * transcript mixes models, costs are summed per-model. Gracefully handles
 * missing usage, corrupt entries, or unreadable files by returning 0,0.
 */
export async function readSessionTotals(transcriptPath) {
  const empty = { sessionTokens: 0, sessionCost: 0 };
  if (typeof transcriptPath !== 'string' || transcriptPath === '') return empty;

  let text;
  try {
    const handle = await open(transcriptPath, 'r');
    try {
      const { size } = await handle.stat();
      const buffer = Buffer.alloc(Math.min(size, TAIL_BYTES * 16)); // read more for full history
      await handle.read(buffer, 0, buffer.length, 0);
      text = buffer.toString('utf8');
    } finally {
      handle.close();
    }
  } catch {
    return empty;
  }

  const lines = text.split('\n');
  let totalTokens = 0;
  let totalCost = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === '') continue;

    let entry;
    try {
      entry = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (entry?.type !== 'assistant') continue;

    const message = entry.message ?? {};
    const usage = message.usage;
    if (!usage || typeof usage !== 'object') continue;

    const model = typeof message.model === 'string' ? message.model : null;
    const rates = ratesFor(model);

    // Each turn's cost: cache reads on context, cache writes on input, output.
    const inputTokens = count(usage.input_tokens);
    const cacheCreation = count(usage.cache_creation_input_tokens);
    const cacheRead = count(usage.cache_read_input_tokens);
    const outputTokens = count(usage.output_tokens);

    const tokens = inputTokens + cacheCreation + cacheRead + outputTokens;
    totalTokens += tokens;

    // Cost: input + cache-write at 1.25×, cache-read at 0.1×, output.
    const inputCost = (inputTokens * rates.input) / 1_000_000;
    const cacheWriteCost = (cacheCreation * rates.cacheWrite) / 1_000_000;
    const cacheReadCost = (cacheRead * rates.cacheRead) / 1_000_000;
    const outputCost = (outputTokens * rates.output) / 1_000_000;

    totalCost += inputCost + cacheWriteCost + cacheReadCost + outputCost;
  }

  return { sessionTokens: totalTokens, sessionCost: totalCost };
}

function ratesFor(model) {
  // Inline the pricing to avoid a circular import. These must match pricing.mjs.
  const id = typeof model === 'string' ? model.toLowerCase() : '';
  const family = id.includes('opus') ? 'opus' : id.includes('haiku') ? 'haiku' : 'sonnet';
  const table = {
    opus: { input: 15, output: 75, cacheWrite: 18.75, cacheRead: 1.5 },
    sonnet: { input: 3, output: 15, cacheWrite: 3.75, cacheRead: 0.3 },
    haiku: { input: 1, output: 5, cacheWrite: 1.25, cacheRead: 0.1 },
  };
  return table[family];
}

function count(value) {
  return Number.isFinite(value) && value > 0 ? value : 0;
}
