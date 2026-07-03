import { describe, it, expect } from 'vitest';
import { NodeOrtBackend } from '../src/ort-backend';

describe('NodeOrtBackend', () => {
  const backend = new NodeOrtBackend();

  it('createTensor for int64 from Int32Array', () => {
    const tensor = backend.createTensor('int64', new Int32Array([1, 2, 3]), [3]);
    expect(tensor.type).toBe('int64');
    expect(tensor.dims).toEqual([3]);
    expect(tensor.data).toBeInstanceOf(BigInt64Array);
  });

  it('createTensor for int64 from number[]', () => {
    const tensor = backend.createTensor('int64', [1, 2, 3], [3]);
    expect(tensor.type).toBe('int64');
    expect(tensor.dims).toEqual([3]);
    expect(tensor.data).toBeInstanceOf(BigInt64Array);
  });

  it('createTensor for float32 from Float32Array', () => {
    const tensor = backend.createTensor('float32', new Float32Array([1.0, 2.0, 3.0]), [3]);
    expect(tensor.type).toBe('float32');
    expect(tensor.dims).toEqual([3]);
    expect(tensor.data).toBeInstanceOf(Float32Array);
  });

  it('createTensor for float32 from number[]', () => {
    const tensor = backend.createTensor('float32', [1.0, 2.0], [2]);
    expect(tensor.type).toBe('float32');
    expect(tensor.dims).toEqual([2]);
  });

  it('dispose tensor does nothing', () => {
    const tensor = backend.createTensor('float32', [1.0, 2.0], [2]);
    expect(() => tensor.dispose()).not.toThrow();
  });
});
