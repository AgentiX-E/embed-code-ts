/**
 * Unit tests for batch-processor — concurrency + progress + timeout.
 */
import { describe, it, expect } from 'vitest';
import { processBatch } from '../../src/batch-processor';

describe('processBatch', () => {
  it('processes all items sequentially with concurrency=1', async () => {
    const results: number[] = [];
    const items = [1, 2, 3, 4, 5];
    await processBatch(items, async (n) => { results.push(n); }, { concurrency: 1 });
    expect(results).toEqual([1, 2, 3, 4, 5]);
  });

  it('processes all items concurrently', async () => {
    const completed: number[] = [];
    const items = ['a', 'b', 'c', 'd'];
    await processBatch(items, async (s) => {
      await new Promise((r) => setTimeout(r, 5));
      completed.push(s.charCodeAt(0));
    }, { concurrency: 4 });
    expect(completed.length).toBe(4);
  });

  it('calls onProgress callback', async () => {
    const progress: number[] = [];
    await processBatch([1, 2, 3], async () => {}, {
      concurrency: 1,
      onProgress: (completed, total) => progress.push(completed),
    });
    expect(progress).toEqual([1, 2, 3]);
  });

  it('handles empty items', async () => {
    await expect(processBatch([], async () => {})).resolves.toBeUndefined();
  });

  it('continues on individual item failure', async () => {
    const results: string[] = [];
    await processBatch(['ok', 'fail', 'ok2'], async (s) => {
      if (s === 'fail') throw new Error('fail');
      results.push(s);
    }, { concurrency: 1 });
    expect(results).toEqual(['ok', 'ok2']);
  });

  it('defaults concurrency when not specified', async () => {
    const results: number[] = [];
    await processBatch([1, 2], async (n) => { results.push(n); });
    expect(results.length).toBe(2);
  });
});
