/**
 * Batch processor — concurrency-controlled batch embedding.
 *
 * Implements backpressure, progress callbacks, and timeout per item.
 */
import type { BatchOptions } from './embedder-interface';

export async function processBatch<T>(
  items: T[],
  processItem: (item: T) => Promise<void>,
  options: BatchOptions = {},
): Promise<void> {
  const concurrency =
    options.concurrency ??
    Math.max(1, Math.floor((typeof navigator !== 'undefined' ? 2 : await importNodeCores()) / 2));
  const timeout = options.timeout ?? 30000;
  let completed = 0;
  const total = items.length;
  let index = 0;

  const workers: Promise<void>[] = [];
  for (let i = 0; i < Math.min(concurrency, total); i++) {
    workers.push(worker());
  }

  async function worker(): Promise<void> {
    while (index < items.length) {
      const i = index++;
      if (i >= items.length) break;
      try {
        await Promise.race([
          processItem(items[i]!),
          new Promise<void>((_, reject) =>
            setTimeout(() => reject(new Error(`Timeout after ${timeout}ms`)), timeout),
          ),
        ]);
      } catch (err) {
        // Individual item failure does not stop batch — caller handles errors
      }
      completed++;
      if (options.onProgress) {
        options.onProgress(completed, total);
      }
    }
  }

  await Promise.all(workers);
}

async function importNodeCores(): Promise<number> {
  try {
    const os = await import('node:os');
    return os.cpus().length;
  } catch {
    return 4; // browser default
  }
}
