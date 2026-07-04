/**
 * Unit tests for int64 conversion utilities.
 */
import { describe, it, expect } from 'vitest';
import { int32ToBigInt64 } from '../../src/int64-utils';

describe('int32ToBigInt64', () => {
  it('converts Int32Array to BigInt64Array', () => {
    const input = new Int32Array([0, 1, 101, 102, 103]);
    const result = int32ToBigInt64(input);
    expect(result).toBeInstanceOf(BigInt64Array);
    expect(result.length).toBe(5);
    expect(result[0]).toBe(BigInt(0));
    expect(result[1]).toBe(BigInt(1));
    expect(result[2]).toBe(BigInt(101));
    expect(result[3]).toBe(BigInt(102));
    expect(result[4]).toBe(BigInt(103));
  });

  it('converts regular number array to BigInt64Array', () => {
    const input = [0, 1, 2, 3];
    const result = int32ToBigInt64(input);
    expect(result).toBeInstanceOf(BigInt64Array);
    expect(result.length).toBe(4);
    expect(result[0]).toBe(BigInt(0));
    expect(result[3]).toBe(BigInt(3));
  });

  it('handles empty arrays', () => {
    const result = int32ToBigInt64([]);
    expect(result).toBeInstanceOf(BigInt64Array);
    expect(result.length).toBe(0);
  });

  it('handles large token IDs', () => {
    const input = new Int32Array([30522, 30523, 30524]);
    const result = int32ToBigInt64(input);
    expect(result.length).toBe(3);
    expect(result[0]).toBe(BigInt(30522));
  });

  it('does not mutate input Int32Array', () => {
    const input = new Int32Array([1, 2, 3]);
    const copy = new Int32Array(input);
    int32ToBigInt64(input);
    expect(input).toEqual(copy);
  });
});
