/**
 * Unit tests for batch-processor — concurrency + progress + timeout.
 */
import { describe, it, expect } from 'vitest';
import { processBatch } from '../../src/batch-processor';

describe('processBatch', () => {
  it('processes all items sequentially with concurrency=1', async () => {
    const results: number[] = [];
    const items = [1, 2, 3, 4, 5];
    await processBatch(
      items,
      async (n) => {
        results.push(n);
      },
      { concurrency: 1 },
    );
    expect(results).toEqual([1, 2, 3, 4, 5]);
  });

  it('processes all items concurrently', async () => {
    const completed: number[] = [];
    const items = ['a', 'b', 'c', 'd'];
    await processBatch(
      items,
      async (s) => {
        await new Promise((r) => setTimeout(r, 5));
        completed.push(s.charCodeAt(0));
      },
      { concurrency: 4 },
    );
    expect(completed.length).toBe(4);
  });

  it('calls onProgress callback', async () => {
    const progress: number[] = [];
    await processBatch([1, 2, 3], async () => {}, {
      concurrency: 1,
      onProgress: (completed, _total) => progress.push(completed),
    });
    expect(progress).toEqual([1, 2, 3]);
  });

  it('handles empty items', async () => {
    const result = await processBatch([], async () => {});
    expect(result.errors).toEqual([]);
  });

  it('continues on individual item failure', async () => {
    const results: string[] = [];
    const result = await processBatch(
      ['ok', 'fail', 'ok2'],
      async (s) => {
        if (s === 'fail') throw new Error('fail');
        results.push(s);
      },
      { concurrency: 1 },
    );
    expect(results).toEqual(['ok', 'ok2']);
    expect(result.errors.length).toBe(1);
    expect(result.errors[0]!.index).toBe(1);
  });

  it('defaults concurrency when not specified', async () => {
    const results: number[] = [];
    await processBatch([1, 2], async (n) => {
      results.push(n);
    });
    expect(results.length).toBe(2);
  });

  it('uses default concurrency from os.cpus()', async () => {
    const results: number[] = [];
    // No concurrency option — triggers importNodeCores() path
    await processBatch([10, 20, 30], async (n) => {
      results.push(n);
    });
    expect(new Set(results)).toEqual(new Set([10, 20, 30]));
  });

  it('times out on slow items', async () => {
    const results: number[] = [];
    await processBatch(
      [1, 2],
      async (n) => {
        if (n === 1) await new Promise((r) => setTimeout(r, 200));
        results.push(n);
      },
      { concurrency: 2, timeout: 50 },
    );
    // Item 1 should have timed out, item 2 should complete
    expect(results).toContain(2);
    expect(results).not.toContain(1);
  });

  it('handles items exactly at concurrency boundary', async () => {
    const results: number[] = [];
    // concurrency=2, items=2: worker loop breaks at i>=2 immediately
    await processBatch([5, 6], async (n) => results.push(n), { concurrency: 3 });
    expect(results).toEqual([5, 6]);
  });
});
