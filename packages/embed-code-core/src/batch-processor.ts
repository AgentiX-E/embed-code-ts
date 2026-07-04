/**
 * Batch processor — concurrency-controlled batch embedding.
 *
 * Implements backpressure, progress callbacks, and timeout per item.
 */
import type { BatchOptions } from './embedder-interface';

export async function processBatch<T>(
  items: T[],
  processItem: (item: T, index: number) => Promise<void>,
  options: BatchOptions = {},
): Promise<{ errors: Array<{ index: number; error: Error }> }> {
  const concurrency =
    options.concurrency ??
    Math.max(1, Math.floor((typeof navigator !== 'undefined' ? 2 : await importNodeCores()) / 2));
  const timeout = options.timeout ?? 30000;
  let completed = 0;
  const total = items.length;
  let index = 0;

  const workers: Promise<{ errors: Array<{ index: number; error: Error }> }>[] = [];
  for (let i = 0; i < Math.min(concurrency, total); i++) {
    workers.push(worker());
  }

  async function worker(): Promise<{ errors: Array<{ index: number; error: Error }> }> {
    const errors: Array<{ index: number; error: Error }> = [];
    while (index < items.length) {
      const i = index++;
      if (i >= items.length) break;
      let timer: ReturnType<typeof setTimeout>;
      try {
        await Promise.race([
          processItem(items[i]!, i),
          new Promise<void>((_, reject) => {
            timer = setTimeout(() => reject(new Error(`Timeout after ${timeout}ms`)), timeout);
          }),
        ]);
      } catch (err) {
        errors.push({ index: i, error: err instanceof Error ? err : new Error(String(err)) });
      } finally {
        clearTimeout(timer!);
      }
      completed++;
      if (options.onProgress) {
        options.onProgress(completed, total);
      }
    }
    return { errors };
  }

  const results = await Promise.all(workers);
  const allErrors: Array<{ index: number; error: Error }> = [];
  for (const r of results) {
    allErrors.push(...r.errors);
  }
  return { errors: allErrors };
}

async function importNodeCores(): Promise<number> {
  try {
    const os = await import('node:os');
    return os.cpus().length;
  } catch {
    return 4; // browser default
  }
}
