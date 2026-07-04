/**
 * Converts Int32Array token IDs to BigInt64Array for ONNX Runtime int64 tensors.
 * Used by both Node.js and Web ONNX backends.
 */
export function int32ToBigInt64(data: Int32Array | number[]): BigInt64Array {
  const arr = data instanceof Int32Array ? new Int32Array(data) : new Int32Array(data);
  const result = new BigInt64Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    result[i] = BigInt(arr[i]!);
  }
  return result;
}
